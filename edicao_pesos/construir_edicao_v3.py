"""Rodada 2: correções de posto baixo por camada, ridge primal com G = KᵀK de todas as posições.

Para cada alvo j da camada L:  w_j = (G_L + λ I)⁻¹ b_j,  b_j = Σ_{posições de erro do alvo j} k_i
(mínimos quadrados fechados com alvo 1 nos pontos de decisão do erro e 0 em todas as
outras ~1,3 M posições). ΔW_L = Σ_j d̂_j w_jᵀ. λ = λ_rel · tr(G)/n_pos. Sem gradiente.
Imprime a resposta w_j·k nos pontos rotulados (erro, alvo 1; vizinhos, alvo 0), com
validação cruzada por instância (ajuste nas pares, leitura nas ímpares, e vice-versa).
"""
import sys, json, torch, collections
from pathlib import Path
ALVOS = {"os1": (23, "2", "5"), "os2": (23, "6", "1"), "ow1": (24, "0", "3"), "b3a": (31, "1", "2"), "b3b": (31, "5", "0")}

def main(pos_path, saida_dir, lambda_rel, peso_viz=0.0, sufixo="v3"):
    D = torch.load(pos_path); cam = D["camadas"]; acts = D["acts"].float(); meta = D["meta"]; n = D["n_pos"]
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    snap = Path(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    def peso(nome):
        with safe_open(snap / idx[nome], "pt") as f: return f.get_tensor(nome).float()
    WU = peso("lm_head.weight"); gamma = peso("model.norm.weight")
    par = torch.tensor([int(m["instance"][3:]) % 2 for m in meta])
    out = Path(saida_dir); out.mkdir(parents=True, exist_ok=True)
    for L in cam:
        G = D["G"][L].cuda(); lam = lambda_rel * G.trace() / n; ci = cam.index(L)
        if peso_viz:   # vizinhos (alvo 0) contam peso_viz vezes mais do que uma posição comum
            Kv = acts[[i for i, m in enumerate(meta) if m["nome"].startswith("viz")], ci].cuda()
            G = G + peso_viz * (Kv.T @ Kv); del Kv
        ci = cam.index(L); alvos = [a for a, (l, _, _) in ALVOS.items() if l == L]
        A = G + lam * torch.eye(G.shape[0], device=G.device)
        Us, Vs = [], []
        for a in alvos:
            _, g, w_ = ALVOS[a]
            iA = [i for i, m in enumerate(meta) if m["nome"] == a]
            # validação: b só com instâncias de um fold (G inclui tudo; aproximação)
            linhas = []
            for fold in (0, 1):
                b = acts[[i for i in iA if par[i] != fold], ci].sum(0).cuda()
                w = torch.linalg.solve(A, b)
                resp = (acts[:, ci].cuda() @ w).cpu(); te = par == fold
                grupos = collections.defaultdict(list)
                for i, m in enumerate(meta):
                    if te[i]: grupos[m["nome"]].append(resp[i].item())
                ref = torch.tensor(grupos[a]).mean()
                linhas.append(f"    fold {fold}: " + " | ".join(f"{k}={torch.tensor(v).mean()/ref:+.2f}±{torch.tensor(v).std()/ref:.2f}" for k, v in sorted(grupos.items()) if len(v) > 1))
            b = acts[iA, ci].sum(0).cuda(); w = torch.linalg.solve(A, b)
            resp = (acts[iA, ci].cuda() @ w).mean().item()
            w = w / resp                                   # resposta média 1 nos pontos de erro
            ids = [tok(t, add_special_tokens=False).input_ids[0] for t in (g, w_)]
            d = (WU[ids[0]] - WU[ids[1]]) * gamma; d = d / d.norm()
            Us.append(d); Vs.append(w.cpu())
            print(f"{a} (L{L}, n={len(iA)}): resposta relativa à média dos erros (alvo 1; vizinhos ~0)\n" + "\n".join(linhas))
        f = out / f"fatores_{sufixo}_L{L}.pt"
        torch.save({"U": torch.stack(Us, 1).contiguous(), "V": torch.stack(Vs, 1).contiguous(), "camada": L, "alvos": alvos, "lambda_rel": lambda_rel}, f)
        print("  salvo", f, "posto", len(alvos))
        del G, A; torch.cuda.empty_cache()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0, sys.argv[5] if len(sys.argv) > 5 else "v3")
