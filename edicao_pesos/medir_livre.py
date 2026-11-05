"""Mede o modelo (editado ou não) nas RESPOSTAS LIVRES dele mesmo, por teacher forcing, nos
pontos de decisão localizados por regex no texto dele (mesmos nomes de classe dos traços).
Guarda x (entrada da MLP) nas CAMADAS em cada marca e, opcionalmente, G_x; formato igual ao
de coletar_residual.py tracos, para servir de positivos ao construtor e de base à calibração.
Uso: medir_livre.py <saida.pt> <a.jsonl,b.jsonl> [config|-] [gpu] [semG]
"""
import sys, json, re, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems, get_check_base_tables, compute_oversize, compute_overweight
from src.interp.experiment import chat_prompt
from edicao_pesos.edicoes import aplicar
MODEL = "Qwen/Qwen3-14B"; CAMADAS = [20, 22, 23, 24, 26, 28, 30, 31, 32, 34]
ITEM_RE = re.compile(r"Item (\d+) \(.*?\):\*?\*?\n", re.S)
ORD = ["First","Second","Third","Fourth","Fifth","Sixth","Seventh","Eighth","Ninth","Tenth"]

def marcas_livres(resp, info, base):
    """[(offset, nome, gold_txt)] no texto da resposta."""
    bags = info["bag_list"][1:]; cls = info["customer_class"]; rt = info["routine"]; out = []
    for m in re.finditer(r"- \**(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)\s*Bag:\**\s*\**\$", resp):
        k = ORD.index(m.group(1))
        if k >= len(base): continue
        g = str(base[k]); nb = f"lista_bag{min(k+1, 4)}"; out.append((m.end(), nb, g[0]))
        if len(g) > 1 and resp[m.end():m.end()+1] == g[0]: out.append((m.end()+1, nb + "_d2", g[1]))
    heads = list(ITEM_RE.finditer(resp))
    for j, h in enumerate(heads):
        k = int(h.group(1)) - 2
        if k < 0 or k >= len(bags): continue
        end = heads[j+1].start() if j+1 < len(heads) else len(resp); body = resp[h.end():end]; off = h.end()
        b = bags[k]; sz = sum(b["size"]); w = b["weight"]; comp = base[k] == 0 and cls in ("Business", "First")
        g_os = compute_oversize(b, rt); g_ow = compute_overweight(b, rt, cls, comp); lim = 70 if comp else 50
        m = re.search(r"Weight: \d+ lbs\.?\n\s*-", body)
        if m: out.append((off + m.end(), "peso_lim", " Over" if w > lim else " Within"))
        m = re.search(r"Over the standard weight limit of (50|70) lbs", body)
        if m and w > lim: out.append((off + m.end(), "peso_faixa", " but" if w <= 70 else " and"))
        m = re.search(r"For dimensions over 6", body)
        if m and sz > 62: out.append((off + m.end(), "os_faixa", "2" if sz <= 65 else "5"))
        m = re.search(r"For dimensions over \d+ inches up to \d+ inches[^\n$]*the fee is \$", body)
        if m and g_os > 0:
            g = str(g_os); out.append((off + m.end(), "os_fee", g[0]))
            if body[m.end():m.end()+1] == g[0]: out.append((off + m.end() + 1, "os_fee_d2", g[1]))
        m = re.search(r"For weights over 5", body)
        if m and 50 < w <= 70 and not comp: out.append((off + m.end(), "ow_faixa_d2", "0" if w <= 53 else "3"))
        m = re.search(r"For weights over \d+ lbs up to \d+ lbs[^\n$]*the fee is \$", body)
        if m and g_ow > 0:
            g = str(g_ow); out.append((off + m.end(), "ow_fee", g[0]))
            if body[m.end():m.end()+1] == g[0]: out.append((off + m.end() + 1, "ow_fee_d2", g[1]))
    return out

def main(out_path, arquivos, cfg_path=None, gpu=0, com_G=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = f"cuda:{gpu}"; tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
    if cfg_path:
        for l in aplicar(model, tok, json.load(open(cfg_path))): print("  edição:", l)
    dm = model.config.hidden_size; store = {}
    for L in CAMADAS:
        model.model.layers[L].mlp.register_forward_pre_hook(lambda m, inp, L=L: store.__setitem__(L, inp[0][0].detach()))
    G = {L: torch.zeros(dm, dm, dtype=torch.float32, device=dev) for L in CAMADAS} if com_G else None
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}; tables = get_check_base_tables()
    acts, meta, n_pos, perdidas = [], [], 0, 0
    with torch.no_grad():
        for arq in arquivos:
            for r in map(json.loads, open(arq)):
                if r.get("finish_reason") == "length": continue
                c, idx = int(r["instance_id"][1]), int(r["instance_id"][3:]); info = probs[c][idx]["info"]
                base = [int(tables[min(3, i)][info["direction"]][info["customer_class"]][info["routine"]]) for i in range(len(info["bag_list"]) - 1)]
                prefix = chat_prompt(tok, r["question"], "one_shot"); full = prefix + r["response"]
                enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                ids = torch.tensor([enc["input_ids"]], device=dev); starts = [a for a, b in enc["offset_mapping"]]
                logits = model(input_ids=ids, use_cache=False).logits[0].float(); n_pos += ids.shape[1]
                if com_G:
                    for L in CAMADAS:
                        x = store[L].float(); G[L] += x.T @ x
                for ch, nome, gtxt in marcas_livres(r["response"], info, base):
                    ch_abs = len(prefix) + ch
                    pos_tok = next((i for i, a in enumerate(starts) if a >= ch_abs), None)
                    if pos_tok is None or starts[pos_tok] != ch_abs: perdidas += 1; continue
                    p_ = pos_tok - 1; gold = tok(gtxt, add_special_tokens=False).input_ids[0]
                    lg = logits[p_]; top = lg.topk(5); outros = lg.clone(); outros[gold] = -1e9
                    acts.append(torch.stack([store[L][p_] for L in CAMADAS]).half().cpu())
                    meta.append(dict(instance=r["instance_id"], origem=Path(arq).stem, nome=nome, gold=gold, gold_txt=gtxt, top=top.indices.tolist(),
                                     top_txt=[tok.decode([t]) for t in top.indices.tolist()], margem=(lg[gold] - outros.max()).item(), traco_ok=False))
                print(Path(arq).stem, r["instance_id"], ids.shape[1], len(acts), flush=True)
    torch.save(dict(camadas=CAMADAS, G={L: G[L].cpu() for L in CAMADAS} if com_G else None, n_pos=n_pos, acts=torch.stack(acts), meta=meta), out_path)
    cnt = collections.defaultdict(lambda: [0, 0])
    for m in meta: k = (m["nome"], m["gold_txt"]); cnt[k][1] += 1; cnt[k][0] += (m["top"][0] == m["gold"])
    for k in sorted(cnt): print(f"{k[0]:12s} {k[1]!r:9s} acerto top-1 {cnt[k][0]:3d}/{cnt[k][1]:3d}")
    print("salvo", out_path, n_pos, "posições,", len(acts), "marcas,", perdidas, "marcas desalinhadas")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2].split(","), None if len(a) < 4 or a[3] == "-" else a[3], int(a[4]) if len(a) > 4 else 0, com_G=not (len(a) > 5 and a[5] == "semG"))
