"""Junta fatias geradas em GPUs diferentes: juntar.py <saida.jsonl> <fatia1.jsonl> <fatia2.jsonl> ..."""
import sys, json, collections
rows = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
ordem = {}
sys.path.insert(0, "."); from src.interp.experiment import sample_problems
for i, s in enumerate(sample_problems(0.3, 42)): ordem[s["instance_id"]] = i
rows.sort(key=lambda r: ordem[r["instance_id"]])
with open(sys.argv[1], "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
acc = sum(r["correct"] for r in rows); por_c = collections.Counter(r["complexity"] for r in rows if r["correct"])
dentro10 = sum(1 for r in rows if r["pred_cents"] and abs(r["pred_cents"]/100 - r["gold"]) <= 0.1*r["gold"]); sem = sum(1 for r in rows if r["pred_cents"] is None)
print(f"RESULTADO {sys.argv[1]}: acertos {acc}/{len(rows)} (C0/C1/C2 = {por_c[0]}/{por_c[1]}/{por_c[2]}), a ±10%: {dentro10}, sem resposta: {sem}")
