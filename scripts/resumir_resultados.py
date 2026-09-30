"""Tabelas-resumo do estudo de informação irrelevante (uma linha por modelo).

Lê results/irrelevant_info/<execução>/ (gerações e, se existirem, capturas)
e grava em results/irrelevant_info/resumo/:

  acuracia.csv          acertos na pergunta limpa, por complexidade, e proximidade do gabarito
  comportamento.csv     taxa de desvio por condição e p (Fisher) contra o controle do cenário
  acertos_desfeitos.csv acertos da pergunta limpa desfeitos no controle e com frase
  ativacoes.csv         convergência, camada da escolha, rank do gabarito e deslocamento do
                        residual (só se as capturas existirem)
  conjunto.json         testes agregados citados no README

As capturas (.pt, ~23 GB) não vão para o repositório público; estas tabelas
guardam o que se tirou delas. Uso: python scripts/resumir_resultados.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interp import summary as S  # noqa: E402
from src.interp.fila import load_queue, run_complete  # noqa: E402

RESULTS = ROOT / "results" / "irrelevant_info"
OUT = RESULTS / "resumo"
P = ["prompt:estava_contando_1+1", "prompt:parei_para_respirar", "prompt:qual_o_valor"]
RS = ["reasoning:estava_contando_1+1", "reasoning:parei_para_respirar", "reasoning:qual_o_valor"]
CONTROLS = ["control", "reasoning:control"]


def runs() -> list[tuple[str, Path, int | None]]:
    """(nome, pasta, camadas) das execuções completas, na ordem da fila, mais a referência RTX 5090."""
    out = []
    ref = RESULTS / "Qwen3-8B_one_shot_frac0.30_seed42"
    if (ref / "generations.jsonl").exists():
        out.append(("Qwen3-8B (RTX 5090, referência)", ref))
    for m in load_queue(ROOT / "fila_modelos.yaml"):
        d = m.run_dir(RESULTS, False)
        if d.exists() and run_complete(d)[0]:
            out.append((m.short + (f" ({m.rotulo})" if m.rotulo else ""), d))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acc, beh, und, act = [], [], [], []
    pooled = {"intermediarios": [0, 0, 0, 0], "gemma": [0, 0, 0, 0]}   # frase: desfeitos, total; controle: desfeitos, total
    for name, d in runs():
        g = S.load_generations(d)
        c = g[g.condition == "clean"].set_index("instance_id")
        err = (c.pred_cents / 100 - c.gold).abs() / c.gold
        acc.append({"modelo": name, "acertos": int(c.correct.sum()), "problemas": len(c),
                    "acuracia": c.correct.mean(),
                    **{f"acertos_c{k}": int(c[c.complexity == k].correct.sum()) for k in (0, 1, 2)},
                    "sem_resposta": int(c.pred_cents.isna().sum()),
                    "a_10pct": (err <= 0.10).mean(), "erro_rel_mediano": err.median()})

        b = S.behavior(g)
        row = {"modelo": name}
        for cond in b.index:
            row[f"desvio[{cond}]"] = b.loc[cond, "taxa_desvio"]
            row[f"n[{cond}]"] = int(b.loc[cond, "n"])
            if not np.isnan(b.loc[cond, "p_vs_controle"]):
                row[f"p[{cond}]"] = b.loc[cond, "p_vs_controle"]
        row["loops[reasoning:estava_contando_1+1]"] = int(b.sem_resposta.get(RS[0], 0))
        beh.append(row)

        ok = g[g.instance_id.map(c.correct).fillna(False).astype(bool)]
        ctrl, fr = ok[ok.condition.isin(CONTROLS)], ok[ok.condition.isin(P + RS)]
        u = {"modelo": name, "acertos_limpos": int(c.correct.sum()),
             "desfeitos_controle": int((~ctrl.correct).sum()), "casos_controle": len(ctrl),
             "desfeitos_frase": int((~fr.correct).sum()), "casos_frase": len(fr)}
        und.append(u)
        if "RTX 5090" not in name and u["acertos_limpos"] > 0:
            key = "gemma" if name.startswith("gemma") else "intermediarios"
            for i, k in enumerate(["desfeitos_frase", "casos_frase", "desfeitos_controle", "casos_controle"]):
                pooled[key][i] += u[k]

        I = S.internals(d)
        if I:
            t, L = I["table"], I["n_layers"]
            r = g[g.scenario == "reasoning"].copy()
            r["dist"] = r.instance_id.map(c.response.str.len()) - r.cut_char
            dd = pd.DataFrame([{"instance_id": x["instance_id"], "condition": k, "desloc": v[-1]}
                               for x in S.load_capture_summaries(d) for k, v in x["resid"].items() if k in RS])
            dd = dd.merge(r[["instance_id", "condition", "dist"]], on=["instance_id", "condition"])
            act.append({"modelo": name, "camadas": L, "capturas": I["n_captures"],
                        "convergencia_rel": t.loc["clean", "convergence"] / L,
                        "camada_escolha_rel": t.loc["clean"].get("camada_escolha", np.nan) / L,
                        "rank_final_mediano_gabarito": t.loc["clean"].get("rank_final_mediano", np.nan),
                        "pct_gabarito_top5": t.loc["clean"].get("pct_top5", np.nan),
                        "desloc_ultima_ruido": t.loc["clean_padded", "shift_last"],
                        "desloc_ultima_prompt": t.loc[[x for x in P if x in t.index], "shift_last"].mean(),
                        "desloc_ultima_raciocinio": t.loc[[x for x in RS if x in t.index], "shift_last"].mean(),
                        "desloc_raciocinio_dist_1a3k_chars": dd[(dd.dist > 1000) & (dd.dist < 3000)].desloc.mean()})
        print(f"ok: {name}")

    pd.DataFrame(acc).to_csv(OUT / "acuracia.csv", index=False, float_format="%.4f")
    pd.DataFrame(beh).to_csv(OUT / "comportamento.csv", index=False, float_format="%.4f")
    pd.DataFrame(und).to_csv(OUT / "acertos_desfeitos.csv", index=False)
    if act:
        pd.DataFrame(act).to_csv(OUT / "ativacoes.csv", index=False, float_format="%.4f")
    agg = {}
    for key, (a, b, x, y) in pooled.items():
        agg[key] = {"desfeitos_frase": a, "casos_frase": b, "desfeitos_controle": x, "casos_controle": y,
                    "p_fisher": fisher_exact([[a, b - a], [x, y - x]])[1] if b and y else None}
    (OUT / "conjunto.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"tabelas em {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
