"""Mede o modelo nos traços-gabarito (teacher forcing) e guarda ativações.

Para cada um dos 90 problemas: prompt 1-shot + traço-gabarito, uma passagem.
Em cada ponto de decisão marcado: token correto, top-5 do modelo, margem
logit(correto) - logit(melhor outro). Guarda a ativação (entrada de down_proj,
fp16) nas CAMADAS e acumula G_L = KᵀK sobre todas as posições (restrição de
alvo zero). Uso: medir_traco.py <saida.pt> [config_edicoes.json|-] [semG|comG] [gpu] [reordenar=0|1]
"""
import sys, json, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems
from src.interp.experiment import sample_problems, chat_prompt
from edicao_pesos.traco_gabarito import traco
from edicao_pesos.edicoes import aplicar
MODEL = "Qwen/Qwen3-14B"; CAMADAS = [20, 22, 24, 26, 28, 30, 32, 34]

def main(out_path, cfg_path=None, com_G=True, gpu=0, reordenar=False):
    dev = f"cuda:{gpu}"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
    if cfg_path:
        for l in aplicar(model, tok, json.load(open(cfg_path))): print("  edição:", l)
    dff = model.config.intermediate_size
    G = {L: torch.zeros(dff, dff, dtype=torch.float32, device=dev) for L in CAMADAS}
    store = {}
    for L in CAMADAS:
        mlp = model.model.layers[L].mlp
        mod = mlp.down_proj.down if hasattr(mlp.down_proj, "down") else mlp.down_proj
        mod.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}
    sample = sample_problems(0.3, 42)
    acts, meta, n_pos = [], [], 0
    with torch.no_grad():
        for s in sample:
            p = probs[s["complexity"]][s["problem_idx"]]
            texto, marcas, total = traco(p, reordenar)
            prefix = chat_prompt(tok, s["question"], "one_shot"); full = prefix + texto
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor([enc["input_ids"]], device=dev); starts = [a for a, b in enc["offset_mapping"]]
            logits = model(input_ids=ids, use_cache=False).logits[0].float()
            T = ids.shape[1]; n_pos += T
            if com_G:
                for L in CAMADAS:
                    x = store[L].float(); G[L] += x.T @ x
            for ch, nome, texto_ok in marcas:
                ch_abs = len(prefix) + ch
                pos_tok = next(i for i, a in enumerate(starts) if a >= ch_abs)   # token que começa em ch_abs
                assert starts[pos_tok] == ch_abs, (nome, texto_ok[:10], full[ch_abs-5:ch_abs+5])
                p_ = pos_tok - 1                                                 # posição que o prevê
                gold = enc["input_ids"][pos_tok]
                lg = logits[p_]; top = lg.topk(5)
                outros = lg.clone(); outros[gold] = -1e9
                acts.append(torch.stack([store[L][p_] for L in CAMADAS]).half().cpu())
                meta.append(dict(instance=s["instance_id"], nome=nome, gold=gold, gold_txt=tok.decode([gold]),
                                 top=top.indices.tolist(), top_txt=[tok.decode([t]) for t in top.indices.tolist()],
                                 margem=(lg[gold] - outros.max()).item(), traco_ok=(total == s["gold"])))
            print(s["instance_id"], T, len(acts), flush=True)
    torch.save(dict(camadas=CAMADAS, G={L: G[L].cpu() for L in CAMADAS} if com_G else None, n_pos=n_pos,
                    acts=torch.stack(acts) if com_G else None, meta=meta), out_path)
    c = collections.defaultdict(lambda: [0, 0])
    for m in meta:
        k = (m["nome"], m["gold_txt"]); c[k][1] += 1; c[k][0] += (m["top"][0] == m["gold"])
    for k in sorted(c): print(f"{k[0]:12s} {k[1]!r:6s} acerto top-1 {c[k][0]:3d}/{c[k][1]:3d}")
    print("salvo", out_path, n_pos, "posições")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2] if len(a) > 2 and a[2] != "-" else None, com_G=not (len(a) > 3 and a[3] == "semG"), gpu=int(a[4]) if len(a) > 4 else 0, reordenar=bool(int(a[5])) if len(a) > 5 else False)
