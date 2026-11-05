"""Edições manuais de pesos (sem treinamento), aplicadas a um modelo carregado.

Config JSON:
{
  "escalas_cabecas": {"L.H": fator, ...},   # multiplica a fatia da cabeça H em o_proj da camada L
  "delta_down_proj": [ {"camada": L, "arquivo": "caminho.pt", "fator": 1.0}, ... ]
      # soma fator*ΔW (d_model x d_ff) a down_proj.weight da camada L (ΔW vem de construir_edicao.py)
  "vies_residual": [ {"camada": L, "token_mais": "2", "token_menos": "5", "alpha": a}, ... ]
      # soma a*(W_U[mais]-W_U[menos])/||.|| à saída do MLP da camada L (bias novo em down_proj)
}
  "edicao_fatorada": [ {"camada": L, "arquivo": "fatores.pt", "fator": a}, ... ]
      # W_down <- W_down + a * U Vᵀ, com U [d_model x r] e V [d_ff x r] guardados em fp32
      # (forma fatorada: evita o arredondamento do bf16, que zera ΔW somado entrada a entrada)
"""
import json, torch


class DownProjEditado(torch.nn.Module):
    """down_proj com o peso efetivo W + U Vᵀ (termo de posto baixo mantido em fp32)."""
    def __init__(self, down, U, V):
        super().__init__()
        self.down = down
        self.register_buffer("U", U.float().to(down.weight.device))   # [d_model, r]
        self.register_buffer("V", V.float().to(down.weight.device))   # [d_ff, r]
    def forward(self, x):
        y = self.down(x)
        return y + ((x.float() @ self.V) @ self.U.T).to(y.dtype)


def aplicar(model, tok, cfg: dict) -> list[str]:
    log = []
    H = model.config.num_attention_heads; D = model.config.head_dim
    for key, f in cfg.get("escalas_cabecas", {}).items():
        L, h = map(int, key.split("."))
        W = model.model.layers[L].self_attn.o_proj.weight
        W.data[:, h*D:(h+1)*D] *= f
        log.append(f"o_proj camada {L} cabeça {h} x {f}")
    for e in cfg.get("delta_down_proj", []):
        down = model.model.layers[e["camada"]].mlp.down_proj
        dW = torch.load(e["arquivo"]).to(down.weight.device, down.weight.dtype) * e.get("fator", 1.0)
        down.weight.data += dW
        log.append(f"down_proj camada {e['camada']}: += {e.get('fator', 1.0)} * ΔW de {e['arquivo']} (||ΔW||={dW.float().norm():.2f})")
    for e in cfg.get("edicao_fatorada", []):
        F = torch.load(e["arquivo"]); a = e.get("fator", 1.0)
        if "fatores" in e: a = torch.tensor(e["fatores"], dtype=torch.float32).view(1, -1)   # um α por coluna (classe)
        mlp = model.model.layers[e["camada"]].mlp
        base = mlp.down_proj.down if isinstance(mlp.down_proj, DownProjEditado) else mlp.down_proj
        U = F["U"] * a
        if isinstance(mlp.down_proj, DownProjEditado):   # acumula com edição anterior na mesma camada
            U = torch.cat([mlp.down_proj.U.cpu(), U], 1); V = torch.cat([mlp.down_proj.V.cpu(), F["V"]], 1)
        else:
            V = F["V"]
        mlp.down_proj = DownProjEditado(base, U, V)
        log.append(f"down_proj camada {e['camada']}: W += {a} * U Vᵀ de {e['arquivo']} (posto {F['U'].shape[1]})")
    for v in cfg.get("vies_residual", []):
        L = v["camada"]
        WU = model.lm_head.weight
        ids = [tok(t, add_special_tokens=False).input_ids[0] for t in (v["token_mais"], v["token_menos"])]
        d = (WU[ids[0]] - WU[ids[1]]).float()
        d = d / d.norm() * v["alpha"]
        down = model.model.layers[L].mlp.down_proj
        if down.bias is None:
            down.bias = torch.nn.Parameter(torch.zeros(down.out_features, dtype=down.weight.dtype, device=down.weight.device))
        down.bias.data += d.to(down.weight.dtype)
        log.append(f"bias down_proj camada {L}: +{v['alpha']} * dir({v['token_mais']} - {v['token_menos']})")
    return log
