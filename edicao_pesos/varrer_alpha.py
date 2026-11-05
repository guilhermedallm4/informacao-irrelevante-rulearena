"""Varre valores de alpha (intensidade das correções fatoradas) nas sondas, com um só carregamento do modelo.
Uso: varrer_alpha.py <sondas> <passo> <json-lista-de-configs> <saida.json> [deslocamento]
config: {"<arquivo de fatores>": alpha, ...}  (alpha multiplica U; ficheiros em results/edicao_pesos/deltas)
"""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edicao_pesos.avaliar_sondas import carregar, avaliar, resumo
from edicao_pesos.edicoes import DownProjEditado

desloc = int(sys.argv[5]) if len(sys.argv) > 5 else 0
probes = [json.loads(l) for l in open(sys.argv[1])][desloc::int(sys.argv[2])]
configs = json.loads(sys.argv[3]); out = sys.argv[4]
model, tok = carregar(None)
arquivos = sorted({k for c in configs for k in c})
edits = {}
for a in arquivos:
    F = torch.load(f"results/edicao_pesos/deltas/{a}")
    mlp = model.model.layers[F["camada"]].mlp
    mlp.down_proj = DownProjEditado(mlp.down_proj, F["U"], F["V"])
    mlp.down_proj.U0 = mlp.down_proj.U.clone()
    edits[a] = mlp.down_proj
resultados = []
for cfg in configs:
    for a, dp in edits.items():
        dp.U.copy_(dp.U0 * float(cfg.get(a, 0.0)))
    res = avaliar(model, tok, probes)
    print(f"\n#### config {cfg}\n{resumo(res)}", flush=True)
    resultados.append(dict(config=cfg, res=res))
    json.dump(resultados, open(out, "w"))
