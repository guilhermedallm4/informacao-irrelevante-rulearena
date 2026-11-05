"""Compara execuções (jsonl de respostas limpas): acertos, aproximação, erro mediano e tipos de erro por item."""
import json, sys, subprocess, collections, statistics
from pathlib import Path
def carregar(p): return {r["instance_id"]: r for r in map(json.loads, open(p))}
def resumo(rows):
    ok = [r for r in rows.values() if r["correct"]]
    pred = {k: r["pred_cents"] / 100 for k, r in rows.items() if r["pred_cents"]}
    rel = [abs(pred[k] - rows[k]["gold"]) / rows[k]["gold"] for k in pred]
    por_c = collections.Counter(r["complexity"] for r in ok)
    return dict(acertos=len(ok), c=(por_c[0], por_c[1], por_c[2]), dentro10=sum(x <= 0.1 for x in rel),
                mediana=statistics.median(rel) * 100, sem=sum(1 for r in rows.values() if not r["pred_cents"]),
                cortadas=sum(1 for r in rows.values() if r["finish_reason"] != "stop"), ids=sorted(r["instance_id"] for r in ok))
if __name__ == "__main__":
    for p in sys.argv[1:]:
        rows = carregar(p); s = resumo(rows)
        print(f"{Path(p).stem}: acertos {s['acertos']}/90 (C0/C1/C2 {s['c']}), a ±10%: {s['dentro10']}, erro mediano {s['mediana']:.1f}%, sem resposta {s['sem']}, cortadas {s['cortadas']}\n   corretos: {s['ids']}")
        out = subprocess.run([sys.executable, "-I", "edicao_pesos/taxonomia_erros.py", p], capture_output=True, text=True).stdout
        print("   " + "\n   ".join(l for l in out.splitlines() if "problemas:" in l))
