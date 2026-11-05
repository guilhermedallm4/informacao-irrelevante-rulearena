"""Avalia as sondas com (ou sem) edições: acurácia token a token por tipo."""
import json, sys, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edicao_pesos.edicoes import aplicar

MODEL = "Qwen/Qwen3-14B"

def carregar(cfg_path=None, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device,
                                                 attn_implementation="sdpa").eval()
    if cfg_path:
        for l in aplicar(model, tok, json.load(open(cfg_path))): print("  edição:", l)
    return model, tok

@torch.no_grad()
def avaliar(model, tok, probes, device="cuda:0"):
    res = []
    for p in probes:
        ids = tok(p["prefix"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        g = tok(p["gold"], add_special_tokens=False).input_ids[0]
        w = tok(p["wrong"], add_special_tokens=False).input_ids[0]
        res.append(dict(kind=p["kind"], gold=p["gold"], instance=p["instance"], item=p.get("item"),
                        diff=(logits[g] - logits[w]).item(), top1=tok.decode([logits.argmax().item()])))
    return res

def resumo(res):
    by = collections.defaultdict(list)
    for r in res: by[(r["kind"], r["gold"])].append(r)
    linhas = []
    for k in sorted(by):
        rs = by[k]
        acc = sum(r["diff"] > 0 for r in rs) / len(rs)
        top = sum(r["top1"] == r["gold"] for r in rs) / len(rs)
        md = sum(r["diff"] for r in rs) / len(rs)
        linhas.append(f"{k[0]:9s} gabarito={k[1]}  n={len(rs):3d}  correto>errado={acc:5.1%}  top1=gabarito={top:5.1%}  dif. média={md:+.2f}")
    return "\n".join(linhas)

if __name__ == "__main__":
    probes_path, out_path = sys.argv[1], sys.argv[2]
    cfg = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    model, tok = carregar(cfg)
    probes = [json.loads(l) for l in open(probes_path)]
    passo = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    probes = probes[::passo]
    res = avaliar(model, tok, probes)
    json.dump(res, open(out_path, "w"))
    print(resumo(res))
