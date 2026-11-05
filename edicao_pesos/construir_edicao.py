"""Constrói a correção de posto baixo em down_proj a partir das ativações medidas.

Para cada tipo de decisão há um grupo A (contextos em que o modelo erra) e um
grupo B (contextos parecidos em que ele acerta). Na camada escolhida:
  k_A, k_B = médias das ativações de neurônios (entrada de down_proj) nos dois grupos
  d        = direção no resíduo que sobe logit(correto_A) - logit(errado_A)
             = (W_U[c] - W_U[e]) * gamma_norm, normalizada
A correção ΔW (d_model x d_ff) é a solução de mínimos quadrados com chaves
K = [k_A, k_B, k_rnd] e alvos V = [alpha*d, 0, 0]:
  ΔW = V (KᵀK)⁻¹ Kᵀ
Assim ΔW k_A = alpha*d (empurra para o correto nos contextos de erro),
ΔW k_B = 0 e ΔW k_rnd = 0 (nada muda nos contextos de acerto nem na média das
posições comuns). alpha é escolhido à mão. Nenhum gradiente, nenhum treino.
"""
import sys, json, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

GRUPOS = {  # nome: (tipo, gabarito do grupo A (erro), gabarito do grupo B (acerto))
    "os_63_65": ("os_tier", "2", "5"),
    "ow_51_53": ("ow_tier2", "0", "3"),
    "bag3_150": ("bag3", "1", "2"),
}

def main(acts_path, saida_dir, camada, alphas: dict, n_rnd=1):
    D = torch.load(acts_path)
    camadas, acts, dla, meta, rnd = D["camadas"], D["acts"].float(), D["dla_mlp"], D["meta"], D["rnd_acts"].float()
    ci = camadas.index(camada)
    from transformers import AutoTokenizer
    from safetensors import safe_open
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download("Qwen/Qwen3-14B", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    def peso(nome):
        with safe_open(snap / idx[nome], "pt") as f: return f.get_tensor(nome).float()
    WU = peso("lm_head.weight"); gamma = peso("model.norm.weight")
    K, V, log = [], [], []
    for nome, (tipo, gA, gB) in GRUPOS.items():
        iA = [i for i, m in enumerate(meta) if m["kind"] == tipo and m["gold"] == gA]
        iB = [i for i, m in enumerate(meta) if m["kind"] == tipo and m["gold"] == gB]
        kA, kB = acts[iA, ci].mean(0), acts[iB, ci].mean(0)
        proj = acts[iA + iB, ci] @ (kA - kB)
        auc = roc_auc_score([1] * len(iA) + [0] * len(iB), proj.numpy())
        ids = [tok(t, add_special_tokens=False).input_ids[0] for t in (gA, gB)]
        d = (WU[ids[0]] - WU[ids[1]]) * gamma; d = d / d.norm()
        a = alphas.get(nome, 0.0)
        K += [kA, kB]; V += [a * d, torch.zeros_like(d)]
        log.append(f"{nome}: camada {camada} nA={len(iA)} nB={len(iB)} AUC(kA-kB)={auc:.3f} |kA|={kA.norm():.1f} |kB|={kB.norm():.1f} cos={torch.cosine_similarity(kA,kB,dim=0):.3f} "
                   f"DLA_MLP média camada (A)={dla[iA, camada].mean():+.2f} alpha={a}")
    for j in range(n_rnd):
        K.append(rnd[ci].mean(0)); V.append(torch.zeros_like(d))
    K = torch.stack(K, 1); V = torch.stack(V, 1)              # K: [d_ff, m], V: [d_model, m]
    dW = V @ torch.linalg.solve(K.T @ K, K.T)                  # [d_model, d_ff]
    out = Path(saida_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(dW.to(torch.bfloat16), out / f"delta_down_proj_L{camada}.pt")
    U_, S_, Vh_ = torch.linalg.svd(dW, full_matrices=False)
    r = int((S_ > 1e-6 * S_[0]).sum())
    torch.save({"U": (U_[:, :r] * S_[:r]).contiguous(), "V": Vh_[:r].T.contiguous(), "camada": camada, "alphas": alphas},
               out / f"fatores_L{camada}.pt")
    log.append(f"fatores salvos: posto {r}")
    chk = dW @ K
    log.append(f"verificação ΔW·K (norma por coluna): {[round(x, 3) for x in chk.norm(dim=0).tolist()]}  ||ΔW||_F={dW.norm():.3f}")
    # efeito colateral: ΔW nas posições aleatórias
    side = (dW @ rnd[ci].T).norm(dim=0)
    log.append(f"|ΔW k| em {rnd.shape[1]} posições aleatórias: média={side.mean():.3f} máx={side.max():.3f} (alvo nos erros: {max(alphas.values())})")
    print("\n".join(log))
    return out / f"delta_down_proj_L{camada}.pt"

if __name__ == "__main__":
    acts_path, saida, camada = sys.argv[1], sys.argv[2], int(sys.argv[3])
    alphas = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    main(acts_path, saida, camada, alphas)
