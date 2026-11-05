"""Rodada 6: correções fechadas de posto 1 em up_proj.

Para a classe j (decisão, token correto) na camada L:
  chave  v_j = A⁻¹ b_j no fluxo residual x (entrada da MLP, 5 120 dim),
         A = G_x + λI + β K_MᵀK_M (G_x: todas as posições dos traços e das respostas livres,
         alvo 0; K_M: marcas, alvo 1 na classe), camada por validação cruzada (AUC vs vizinhos);
  padrão u_j sobre os neurônios: minimiza ||W_down diag(ḡ) u − d̂||² + μ||u||², com
         ḡ = média de silu(W_gate x_i) na classe (gate média):  u = ḡ ⊙ W_downᵀ (W_down diag(ḡ²) W_downᵀ + μI)⁻¹ d̂.
Efeito na posição i: W_down (g_i ⊙ u_j) · (v_j·x_i), i.e., só onde a gate está aberta.
u_j é normalizado para projeção média 1 em d̂ na classe; v_j para resposta média 1.
Uso: construir_edicao_up.py <res_tracos.pt> <saida_dir> [λ_rel=0.1] [β=100] [min_erros=2] [sufixo=up1] [G_extra=a.pt,b.pt] [peso_extra=1] [μ_rel=0.01] [excluir=L,L] [gpu=0] [prefixo_classe]
"""
import sys, json, torch, collections
from pathlib import Path
from sklearn.metrics import roc_auc_score

def main(res_path, saida_dir, lambda_rel=0.1, beta=100.0, min_erros=2, sufixo="up1", G_extra="", peso_extra=1.0, mu_rel=0.01, excluir="", dev="cuda:0", prefixo=""):
    excl = {int(x) for x in excluir.split(",") if x}
    D = torch.load(res_path); cam = D["camadas"]; acts = D["acts"]; meta = D["meta"]; n = D["n_pos"]
    for ge in [g for g in G_extra.split(",") if g]:
        E = torch.load(ge); assert E["camadas"] == cam
        if E.get("G") is not None:
            for L in cam: D["G"][L] += peso_extra * E["G"][L]
            n += int(peso_extra * E["n_pos"])
        if "acts" in E and E["acts"] is not None:     # marcas extra (respostas livres) como positivos
            acts = torch.cat([acts, E["acts"]]); meta = meta + E["meta"]; print("marcas extra", ge, len(E["meta"]))
        print("G extra", ge, E["n_pos"], "posições, peso", peso_extra); del E
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    def peso(nome):
        with safe_open(snap / idx[nome], "pt") as f: return f.get_tensor(nome).float()
    WU = peso("lm_head.weight"); gamma = peso("model.norm.weight")
    por = collections.defaultdict(list)
    for i, m in enumerate(meta): por[(m["nome"], m["gold"])].append(i)
    classes = []
    for (nome, g), ix in sorted(por.items()):
        erros = [i for i in ix if meta[i]["top"][0] != g]
        if len(erros) < min_erros or (prefixo and not nome.startswith(prefixo)): continue
        t = collections.Counter(meta[i]["top"][0] for i in erros).most_common(1)[0][0]
        viz = [i for i, m in enumerate(meta) if m["nome"] == nome and m["gold"] != g]
        classes.append(dict(nome=nome, gold=g, gold_txt=meta[ix[0]]["gold_txt"], wrong=t,
                            wrong_txt=next((meta[i]["top_txt"][meta[i]["top"].index(t)] for i in erros if t in meta[i]["top"]), "?"),
                            n=len(ix), n_erros=len(erros), margem_media=sum(meta[i]["margem"] for i in ix) / len(ix),
                            margem_min=min(meta[i]["margem"] for i in ix), ix=ix, viz=viz))
    print(f"{len(classes)} classes com ≥{min_erros} erros")
    par = torch.tensor([int(m["instance"][3:]) % 2 for m in meta]); out = Path(saida_dir); out.mkdir(parents=True, exist_ok=True)
    for ci, L in enumerate(cam):
        if L in excl: continue
        K = acts[:, ci].float().to(dev); G = D["G"][L].to(dev); lam = lambda_rel * G.trace() / n
        for fold in (0, 1, None):
            tr = torch.ones(len(meta), dtype=torch.bool) if fold is None else (par != fold); Kt = K[tr.to(dev)]
            A = G + lam * torch.eye(G.shape[0], device=dev) + beta * (Kt.T @ Kt)
            B = torch.stack([beta * K[[i for i in c["ix"] if tr[i]]].sum(0) for c in classes], 1)
            W = torch.linalg.solve(A, B); R = (K @ W).cpu()
            for j, c in enumerate(classes):
                if fold is None: c.setdefault("v", {})[L] = W[:, j].cpu(); continue
                te = (par == fold); ia = [i for i in c["ix"] if te[i]]; iv = [i for i in c["viz"] if te[i]]
                if not ia: continue
                ra = R[ia, j]; rv = R[iv, j] if iv else torch.zeros(0); ref = ra.mean().item()
                auc = roc_auc_score([1] * len(ia) + [0] * len(iv), torch.cat([ra, rv]).numpy()) if len(iv) else 1.0
                outras = R[[i for i in range(len(meta)) if te[i] and i not in c["ix"] and i not in c["viz"]], j]
                c.setdefault("cv", {}).setdefault(L, []).append(dict(auc=auc, viz=(rv.mean().item() / ref if len(rv) else 0.0), outras=(outras.abs().mean().item() / ref)))
        del K, G; torch.cuda.empty_cache(); print("camada", L, "chaves prontas", flush=True)
    # padrão u por classe na camada escolhida
    por_camada = collections.defaultdict(lambda: ([], [], [])); resumo = []; cache = {}
    for c in classes:
        melhor = max(((sum(x["auc"] for x in cvs) / len(cvs), -abs(sum(x["viz"] for x in cvs) / len(cvs)), L) for L, cvs in c["cv"].items()))
        auc, viz, L = melhor[0], -melhor[1], melhor[2]; ci = cam.index(L)
        if L not in cache:
            cache[L] = (peso(f"model.layers.{L}.mlp.down_proj.weight").to(dev), peso(f"model.layers.{L}.mlp.gate_proj.weight").to(dev))
        Wd, Wg = cache[L]
        X = acts[c["ix"], ci].float().to(dev); Gt = torch.nn.functional.silu(X @ Wg.T)        # [n_c, d_ff]
        gm = Gt.mean(0)
        d = (WU[c["gold"]] - WU[c["wrong"]]) * gamma; d = (d / d.norm()).to(dev)
        M = (Wd * (gm ** 2)) @ Wd.T; mu = mu_rel * M.trace() / M.shape[0]
        y = torch.linalg.solve(M + mu * torch.eye(M.shape[0], device=dev), d); u = gm * (Wd.T @ y)
        cvec = u * (Wd.T @ d)                                   # projeção em d̂ = g·cvec
        proj = Gt @ cvec; E = (Gt * u) @ Wd.T                   # efeito por posição [n_c, d_model]
        cos = torch.nn.functional.cosine_similarity(E, d.expand_as(E), dim=1)
        esc = proj.mean().item(); u = u / esc; cvec = cvec / esc
        v = c["v"][L].to(dev); resp = (X @ v).mean().item(); v = v / resp
        # empurrão efetivo nos vizinhos (mesma decisão, outro token): (v·x)(g·cvec), relativo a 1
        Xv = acts[c["viz"], ci].float().to(dev) if c["viz"] else None
        push_viz = (((Xv @ v) * (torch.nn.functional.silu(Xv @ Wg.T) @ cvec)).mean().item()) if Xv is not None else 0.0
        Us, Vs, nomes = por_camada[L]; Us.append(u.cpu()); Vs.append(v.cpu()); nomes.append(f"{c['nome']}:{c['gold_txt']!r}")
        resumo.append(dict(nome=c["nome"], gold_txt=c["gold_txt"], wrong_txt=c["wrong_txt"], n=c["n"], n_erros=c["n_erros"], margem_media=round(c["margem_media"], 2),
                           margem_min=round(c["margem_min"], 2), camada=L, auc_cv=round(auc, 3), resp_vizinhos=round(viz, 2), push_vizinhos=round(push_viz, 3),
                           cos_medio=round(cos.mean().item(), 3), proj_cv=round((proj.std() / proj.mean()).item(), 3), indice_na_camada=len(Us) - 1, tipo="up"))
        print(f"{c['nome']:14s} {c['gold_txt']!r:9s} erro→{c['wrong_txt']!r:8s} n={c['n']:3d} erros={c['n_erros']:3d} margem={c['margem_media']:+6.2f}  L{L} AUC_cv={auc:.3f} viz={viz:+.2f} push_viz={push_viz:+.3f} cos={cos.mean():.3f} proj_cv={proj.std()/proj.mean():.2f}")
    for L, (Us, Vs, nomes) in por_camada.items():
        torch.save({"U": torch.stack(Us, 1).contiguous(), "V": torch.stack(Vs, 1).contiguous(), "camada": L, "classes": nomes, "tipo": "up"}, out / f"fatores_{sufixo}_L{L}.pt")
    json.dump(resumo, open(out / f"classes_{sufixo}.json", "w"), indent=1); print("salvo em", out)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else 0.1, float(a[4]) if len(a) > 4 else 100.0, int(a[5]) if len(a) > 5 else 2, a[6] if len(a) > 6 else "up1",
         a[7] if len(a) > 7 else "", float(a[8]) if len(a) > 8 else 1.0, float(a[9]) if len(a) > 9 else 0.01, a[10] if len(a) > 10 else "",
         f"cuda:{a[11]}" if len(a) > 11 else "cuda:0", a[12] if len(a) > 12 else "")
