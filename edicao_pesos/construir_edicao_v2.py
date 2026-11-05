"""Correção de posto 1 em down_proj com direção de disparo ajustada por mínimos quadrados fechados.

ΔW = d̂ wᵀ, onde
  d̂ = direção do resíduo que sobe logit(correto) - logit(errado) (W_U[c]-W_U[e]) ⊙ γ, unitária;
  w  = solução de ridge em forma dual sobre TODAS as chaves medidas na camada:
       y_i = 1 nos contextos de erro do grupo, 0 nas demais sondas e em posições aleatórias,
       w = Kᵀ (K Kᵀ + λ I)⁻¹ y,  λ = lambda_rel · média(‖k‖²)   (escolhido à mão).
Assim ΔW k ≈ α d̂ nos contextos de erro e ≈ 0 nos outros. Sem gradiente, sem iterações.
Validação: ajusta nas instâncias pares e mede nas ímpares (e vice-versa).
"""
import sys, json, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
GRUPOS = {"os_63_65": ("os_tier", "2", "5"), "ow_51_53": ("ow_tier2", "0", "3"), "bag3_150": ("bag3", "1", "2")}

def ridge_dual(K, y, lam):
    G = K @ K.T
    return K.T @ torch.linalg.solve(G + lam * torch.eye(len(y)), y)

def main(acts_path, saida_dir, camada, grupo, lambda_rel):
    D = torch.load(acts_path); cam, acts, meta, rnd = D["camadas"], D["acts"].float(), D["meta"], D["rnd_acts"].float()
    ci = cam.index(camada); tipo, gA, gB = GRUPOS[grupo]
    K = torch.cat([acts[:, ci], rnd[ci]], 0)                     # [n, d_ff]
    y = torch.tensor([1.0 if (m["kind"] == tipo and m["gold"] == gA) else 0.0 for m in meta] + [0.0] * rnd.shape[1])
    lam = lambda_rel * (K.pow(2).sum(1).mean())
    # validação cruzada por instância (pares / ímpares)
    inst = [int(m["instance"][3:]) % 2 for m in meta] + [i % 2 for i in range(rnd.shape[1])]
    inst = torch.tensor(inst)
    rel = []
    for fold in (0, 1):
        tr, te = inst != fold, inst == fold
        w = ridge_dual(K[tr], y[tr], lam)
        p = K[te] @ w; yt = y[te]
        kinds = [m["kind"] + m["gold"] for m in meta] + ["rnd"] * rnd.shape[1]
        kt = [k for k, t in zip(kinds, te.tolist()) if t]
        def stat(mask): 
            x = p[mask]; return f"média={x.mean():+.2f} p10={x.quantile(.1):+.2f} p90={x.quantile(.9):+.2f}"
        mA = yt == 1; mB = torch.tensor([k == tipo + gB for k in kt]); mR = torch.tensor([k == "rnd" for k in kt]); mO = ~(mA | mB | mR)
        rel.append(f"  fold {fold}: erros(alvo 1) {stat(mA)} | acertos vizinhos(alvo 0) {stat(mB)} | outras sondas {stat(mO)} | aleatórias {stat(mR)} | AUC erro×vizinho={roc_auc_score(yt[mA|mB].numpy(), p[mA|mB].numpy()):.3f}")
    w = ridge_dual(K, y, lam)
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    snap = Path(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    def peso(nome):
        with safe_open(snap / idx[nome], "pt") as f: return f.get_tensor(nome).float()
    WU = peso("lm_head.weight"); gamma = peso("model.norm.weight")
    ids = [tok(t, add_special_tokens=False).input_ids[0] for t in (gA, gB)]
    d = (WU[ids[0]] - WU[ids[1]]) * gamma; d = d / d.norm()
    out = Path(saida_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / f"fatores_v2_{grupo}_L{camada}.pt"
    torch.save({"U": d.view(-1, 1).contiguous(), "V": w.view(-1, 1).contiguous(), "camada": camada, "grupo": grupo, "lambda_rel": lambda_rel}, f)
    print(f"{grupo} L{camada} λ_rel={lambda_rel} (n chaves={len(y)}, ‖w‖={w.norm():.4f})\n" + "\n".join(rel) + f"\n  salvo {f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
