"""Calibra α por classe: escreve config uniforme (passo 1) ou, dado o traço medido com α0,
calcula α_j = clip((alvo − margem_mín_j)/inclinação_j) e escreve o config calibrado (passo 2)."""
import sys, json, torch, collections
from pathlib import Path
D = "results/edicao_pesos/deltas/"
def classes(sufixo): return json.load(open(f"{D}classes_{sufixo}.json"))
def escrever(cfg_path, alphas, sufixo):
    cls = classes(sufixo); por = collections.defaultdict(dict)
    for j, c in enumerate(cls): por[c["camada"]][c["indice_na_camada"]] = alphas[j]
    ed = [{"camada": L, "arquivo": f"{D}fatores_{sufixo}_L{L}.pt", "fatores": [v[i] for i in sorted(v)]} for L, v in sorted(por.items())]
    json.dump({"edicao_fatorada": ed}, open(cfg_path, "w"), indent=1)
def margens(traco_path):
    meta = torch.load(traco_path)["meta"]; por = collections.defaultdict(list)
    for m in meta: por[(m["nome"], m["gold"])].append(m)
    return por
if __name__ == "__main__":
    modo = sys.argv[1]; sufixo = sys.argv[2] if len(sys.argv) > 2 else "v5"
    cls = classes(sufixo)
    if modo == "uniforme":
        a0 = float(sys.argv[3]); escrever(sys.argv[4], [a0] * len(cls), sufixo); print("config uniforme", a0, "->", sys.argv[4])
    else:   # calibrar <sufixo> <traco_base.pt> <traco_a0.pt> <a0> <alvo> <alpha_max> <saida.json>
        base, med, a0, alvo, amax, out = sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
        mb, mm = margens(base), margens(med); alphas = []
        for c in cls:
            k = (c["nome"], [m for m in mb[(c["nome"], g)] for g in [None]] and None) if False else None
            key = next(kk for kk in mb if kk[0] == c["nome"] and mb[kk][0]["gold_txt"] == c["gold_txt"])
            b = torch.tensor([m["margem"] for m in mb[key]]); a = torch.tensor([m["margem"] for m in mm[key]])
            slope = ((a - b).mean() / a0).item()
            need = alvo - b.quantile(0.05).item()
            alpha = 0.0 if slope <= 1e-4 else min(amax, max(0.0, need / slope))
            alphas.append(round(alpha, 1))
            print(f"{c['nome']:14s} {c['gold_txt']!r:9s} margem base média {b.mean():+6.2f} p5 {b.quantile(0.05):+6.2f} | com α0: média {a.mean():+6.2f} | inclinação {slope:+.4f}/α | α={alpha:.0f}")
        escrever(out, alphas, sufixo); print("config calibrado ->", out)
