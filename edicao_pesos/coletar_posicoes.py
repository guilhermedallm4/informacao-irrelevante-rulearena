"""Rodada 2: estatísticas de TODAS as posições + ativações rotuladas nos pontos de decisão.

Para cada resposta (linha de base e modelo editado), uma passagem com hooks na
entrada de down_proj das camadas 23, 24 e 31:
  - G_L += KᵀK sobre todas as posições (prompt e resposta): restrição de
    "não mudar nada" fora dos pontos de decisão (alvo zero);
  - nos pontos de decisão rotulados, guarda a ativação (fp16) e o rótulo.
Alvos (nome: camada, token correto − token errado):
  os1: L23 "For dimensions over 6"→"2" vs "5" (63–65 in)
  os2: L23 "For dimensions over 62 inches up to "→"6" vs "1" (63–65 in)
  ow1: L24 "For weights over 5"→"0" vs "3" (51–53 lb, mala não gratuita)
  b3a: L31 "Third Bag: $"→"1" vs "2" (3ª mala a $150)
  b3b: L31 "Third Bag: $1"→"5" vs "0" (idem)
Vizinhos guardados com alvo zero (só para validação): >65 in, 54–70 lb,
"For weights over " (→"5"), outras linhas "Bag: $", 3ª mala a $200.
Também escreve sondas_v2.jsonl com prefixos para avaliação token a token.
"""
import json, re, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems, get_check_base_tables
from src.interp.experiment import chat_prompt
MODEL = "Qwen/Qwen3-14B"; CAMADAS = [23, 24, 31]
ITEM_RE = re.compile(r"Item (\d+) \(.*?\):\*?\*?\n", re.S)

def rotulos(resp, info, base):
    """Lista de (char_offset_do_token_alvo, nome, gold, wrong, alvo(1/0))."""
    bags = info["bag_list"][1:]; out = []
    for m in re.finditer(r"- (First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth) Bag: \$", resp):
        if m.group(1) == "Third" and len(base) >= 3 and base[2] in (150, 200):
            if base[2] == 150:
                out.append((m.end(), "b3a", "1", "2", 1))
                if resp[m.end():m.end()+1] == "1": out.append((m.end()+1, "b3b", "5", "0", 1))
            else: out.append((m.end(), "viz_b3_200", "2", "1", 0))
        elif m.group(1) == "Fourth" and len(base) >= 4 and base[3] == 200:
            out.append((m.end(), "viz_b4", "2", "1", 0))
        else:
            out.append((m.end(), "viz_bagX", "", "", 0))
    heads = list(ITEM_RE.finditer(resp))
    for j, h in enumerate(heads):
        k = int(h.group(1)) - 2
        if k < 0 or k >= len(bags): continue
        end = heads[j+1].start() if j+1 < len(heads) else len(resp); body = resp[h.end():end]; off = h.end()
        b = bags[k]; sz = sum(b["size"]); w = b["weight"]
        comp = base[k] == 0 and info["customer_class"] in ("Business", "First")
        m = re.search(r"For dimensions over 6", body)
        if m and sz > 62:
            out.append((off+m.end(), "os1" if sz <= 65 else "viz_os_65", "2" if sz <= 65 else "5", "5" if sz <= 65 else "2", 1 if sz <= 65 else 0))
        m = re.search(r"For dimensions over 62 inches up to ", body)
        if m and 62 < sz <= 65: out.append((off+m.end(), "os2", "6", "1", 1))
        m = re.search(r"For weights over ", body)
        if m and w > 50 and not comp:
            out.append((off+m.end(), "viz_ow_first", "5" if w <= 70 else "7", "0", 0))
            if w <= 70 and resp[off+m.end():off+m.end()+1] == "5":
                out.append((off+m.end()+1, "ow1" if w <= 53 else "viz_ow_70", "0" if w <= 53 else "3", "3" if w <= 53 else "0", 1 if w <= 53 else 0))
    return out

def main(arquivos, out_path, sondas_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    dff = model.config.intermediate_size
    G = {L: torch.zeros(dff, dff, dtype=torch.float32, device="cuda:0") for L in CAMADAS}
    n_pos = 0; store = {}
    for L in CAMADAS:
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}; tables = get_check_base_tables()
    acts, meta, sondas = [], [], []
    with torch.no_grad():
        for arq in arquivos:
            for r in map(json.loads, open(arq)):
                c, idx = int(r["instance_id"][1]), int(r["instance_id"][3:]); info = probs[c][idx]["info"]
                bags = info["bag_list"][1:]
                base = [int(tables[min(3, i)][info["direction"]][info["customer_class"]][info["routine"]]) for i in range(len(bags))]
                prefix = chat_prompt(tok, r["question"], "one_shot"); text = prefix + r["response"]
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                ids = torch.tensor([enc["input_ids"]], device="cuda:0"); offs = enc["offset_mapping"]
                model(input_ids=ids, use_cache=False)
                T = ids.shape[1]; n_pos += T
                for L in CAMADAS:
                    x = store[L].float(); G[L] += x.T @ x
                starts = [a for a, b in offs]
                for ch, nome, gold, wrong, alvo in rotulos(r["response"], info, base):
                    ch_abs = len(prefix) + ch
                    # posição que prevê o token que começa em ch_abs = último token do prefixo
                    p = max(i for i, a in enumerate(starts) if a < ch_abs)
                    acts.append(torch.stack([store[L][p] for L in CAMADAS]).half().cpu())
                    meta.append(dict(instance=r["instance_id"], origem=Path(arq).stem, nome=nome, gold=gold, wrong=wrong, alvo=alvo))
                    if gold: sondas.append(dict(instance=r["instance_id"], origem=Path(arq).stem, kind=nome, gold=gold, wrong=wrong,
                                                prefix=text[:ch_abs], info=""))
                print(r["instance_id"], Path(arq).stem, T, len(acts), flush=True)
    torch.save(dict(camadas=CAMADAS, G={L: G[L].cpu() for L in CAMADAS}, n_pos=n_pos, acts=torch.stack(acts), meta=meta), out_path)
    Path(sondas_path).write_text("\n".join(json.dumps(s) for s in sondas))
    import collections; print(collections.Counter(m["nome"] for m in meta)); print("salvo", out_path, n_pos, "posições")

if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2], sys.argv[3])
