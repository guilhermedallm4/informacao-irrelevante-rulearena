"""Rodada 3: correções fechadas para TODAS as decisões erradas dos traços-gabarito.

Para cada classe j = (decisão, token correto) com erros nos traços:
  t_j  = token errado mais comum (top-1 do modelo quando erra)
  d̂_j  = (W_U[g] − W_U[t_j]) ⊙ γ, unitário
  w_j  = A_L⁻¹ b_j,  A_L = G_L + λI + β K_Mᵀ K_M,  b_j = β Σ_{i∈j} k_i
         (G_L: todas as posições, alvo 0; K_M: posições marcadas, alvo 1 na classe j e 0 nas
          demais, com peso β; mínimos quadrados fechados, sem gradiente)
  camada L escolhida por validação cruzada (problemas pares/ímpares): maior AUC entre a
  resposta nas posições da classe e nas posições vizinhas (mesma decisão, outro token).
Saída: fatores_v5_L{L}.pt por camada (U = d̂_j, V = w_j normalizado p/ resposta média 1) e
classes_v5.json com margens medidas, para calibrar α por classe.
"""
import sys, json, torch, collections
from pathlib import Path
from sklearn.metrics import roc_auc_score

def main(traco_path, saida_dir, lambda_rel=0.1, beta=100.0, min_erros=2, sufixo="v5", G_extra="", peso_extra=1.0, dev="cuda:0"):
    D = torch.load(traco_path); cam = D["camadas"]; acts = D["acts"]; meta = D["meta"]; n = D["n_pos"]
    # Rodada 4: G das respostas livres (alvo zero nos contextos que o modelo escreve sozinho)
    for ge in [g for g in G_extra.split(",") if g]:
        E = torch.load(ge); assert E["camadas"] == cam
        for L in cam: D["G"][L] += peso_extra * E["G"][L]
        n += int(peso_extra * E["n_pos"]); print("G extra", ge, E["n_pos"], "posições, peso", peso_extra)
        del E
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    def peso(nome):
        with safe_open(snap / idx[nome], "pt") as f: return f.get_tensor(nome).float()
    WU = peso("lm_head.weight"); gamma = peso("model.norm.weight")
    # classes
    por = collections.defaultdict(list)
    for i, m in enumerate(meta): por[(m["nome"], m["gold"])].append(i)
    classes = []
    for (nome, g), ix in sorted(por.items()):
        erros = [i for i in ix if meta[i]["top"][0] != g]
        if len(erros) < min_erros: continue
        t = collections.Counter(meta[i]["top"][0] for i in erros).most_common(1)[0][0]
        viz = [i for i, m in enumerate(meta) if m["nome"] == nome and m["gold"] != g]
        classes.append(dict(nome=nome, gold=g, gold_txt=meta[ix[0]]["gold_txt"], wrong=t, wrong_txt=meta[erros[0]]["top_txt"][meta[erros[0]]["top"].index(t)] if t in meta[erros[0]]["top"] else "?",
                            n=len(ix), n_erros=len(erros), margem_media=sum(meta[i]["margem"] for i in ix) / len(ix),
                            margem_min=min(meta[i]["margem"] for i in ix), ix=ix, viz=viz))
    print(f"{len(classes)} classes com ≥{min_erros} erros")
    par = torch.tensor([int(m["instance"][3:]) % 2 for m in meta])
    out = Path(saida_dir); out.mkdir(parents=True, exist_ok=True)
    escolha = {}
    for ci, L in enumerate(cam):
        K = acts[:, ci].float().to(dev)                                  # [n_marcas, dff]
        G = D["G"][L].to(dev); lam = lambda_rel * G.trace() / n
        for fold in (0, 1, None):                                        # None = ajuste final com tudo
            tr = torch.ones(len(meta), dtype=torch.bool) if fold is None else (par != fold)
            Kt = K[tr.to(dev)]
            A = G + lam * torch.eye(G.shape[0], device=dev) + beta * (Kt.T @ Kt)
            Lc = torch.linalg.cholesky(A); del A
            B = torch.stack([beta * K[[i for i in c["ix"] if tr[i]]].sum(0) for c in classes], 1)  # [dff, nc]
            W = torch.cholesky_solve(B, Lc); del Lc
            R = (K @ W).cpu()                                            # resposta de cada marca a cada w_j
            for j, c in enumerate(classes):
                if fold is None:
                    c.setdefault("w", {})[L] = W[:, j].cpu(); continue
                te = (par == fold)
                ia = [i for i in c["ix"] if te[i]]; iv = [i for i in c["viz"] if te[i]]
                if not ia: continue
                ra = R[ia, j]; rv = R[iv, j] if iv else torch.zeros(0)
                ref = ra.mean().item()
                auc = roc_auc_score([1] * len(ia) + [0] * len(iv), torch.cat([ra, rv]).numpy()) if len(iv) and len(set([1] * len(ia) + [0] * len(iv))) > 1 else 1.0
                outras = R[[i for i in range(len(meta)) if te[i] and i not in c["ix"] and i not in c["viz"]], j]
                c.setdefault("cv", {}).setdefault(L, []).append(dict(auc=auc, viz=(rv.mean().item() / ref if len(rv) else 0.0), outras=(outras.abs().mean().item() / ref)))
        del K, G; torch.cuda.empty_cache()
        print("camada", L, "pronta", flush=True)
    # escolha da camada por classe e normalização
    por_camada = collections.defaultdict(lambda: ([], [], []))
    resumo = []
    for c in classes:
        melhor = None
        for L, cvs in c["cv"].items():
            auc = sum(x["auc"] for x in cvs) / len(cvs); viz = sum(x["viz"] for x in cvs) / len(cvs); outras = sum(x["outras"] for x in cvs) / len(cvs)
            cand = (auc, -abs(viz), L)
            if melhor is None or cand > melhor[0]: melhor = (cand, L, auc, viz, outras)
        _, L, auc, viz, outras = melhor
        w = c["w"][L]; ci = cam.index(L)
        resp = (acts[c["ix"], ci].float() @ w).mean().item(); w = w / resp
        ids = (c["gold"], c["wrong"]); d = (WU[ids[0]] - WU[ids[1]]) * gamma; d = d / d.norm()
        Us, Vs, nomes = por_camada[L]; Us.append(d); Vs.append(w); nomes.append(f"{c['nome']}:{c['gold_txt']!r}")
        resumo.append(dict(nome=c["nome"], gold_txt=c["gold_txt"], wrong_txt=c["wrong_txt"], n=c["n"], n_erros=c["n_erros"],
                           margem_media=round(c["margem_media"], 2), margem_min=round(c["margem_min"], 2), camada=L, auc_cv=round(auc, 3),
                           resp_vizinhos=round(viz, 2), resp_outras=round(outras, 3), indice_na_camada=len(Us) - 1))
        print(f"{c['nome']:14s} {c['gold_txt']!r:9s} erro→{c['wrong_txt']!r:8s} n={c['n']:3d} erros={c['n_erros']:3d} margem={c['margem_media']:+6.2f} (mín {c['margem_min']:+6.2f})  L{L} AUC_cv={auc:.3f} viz={viz:+.2f} outras={outras:.3f}")
    for L, (Us, Vs, nomes) in por_camada.items():
        torch.save({"U": torch.stack(Us, 1).contiguous(), "V": torch.stack(Vs, 1).contiguous(), "camada": L, "classes": nomes}, out / f"fatores_{sufixo}_L{L}.pt")
    json.dump(resumo, open(out / f"classes_{sufixo}.json", "w"), indent=1)
    print("salvo em", out)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else 0.1, float(a[4]) if len(a) > 4 else 100.0, int(a[5]) if len(a) > 5 else 2, a[6] if len(a) > 6 else "v5",
         a[7] if len(a) > 7 else "", float(a[8]) if len(a) > 8 else 1.0)
