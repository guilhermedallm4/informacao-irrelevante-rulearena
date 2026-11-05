"""Avalia um ciclo: acertos, ±10%, erro mediano, respostas cortadas e laços (repetição de n-gramas
no fim da resposta mesmo sem cortar). Escreve os ids em laço em <saida>.laco.json.
Uso: avaliar_ciclo.py <respostas.jsonl> [n_gram=12] [min_rep=6]"""
import sys, json, statistics, collections
def em_laco(texto, janela=400, frac=0.9):
    """Laço = o fim da resposta é periódico: para algum período p ≤ 150 palavras, ≥90% das
    últimas `janela` palavras repetem a palavra p posições antes."""
    toks = texto.split()[-(janela + 150):]
    if len(toks) < janela + 150: return False
    fim = toks[-janela:]
    for p in range(1, 151):
        ig = sum(1 for i in range(janela) if fim[i] == toks[len(toks) - janela + i - p])
        if ig / janela >= frac: return True
    return False
def main(path, n=12, min_rep=6):
    rows = [json.loads(l) for l in open(path)]
    acc = sum(r["correct"] for r in rows); pc = collections.Counter(r["complexity"] for r in rows if r["correct"])
    d10 = sum(1 for r in rows if r["pred_cents"] and abs(r["pred_cents"]/100 - r["gold"]) <= 0.1*r["gold"])
    errs = [abs(r["pred_cents"]/100 - r["gold"])/r["gold"] for r in rows if r["pred_cents"]]
    cort = [r["instance_id"] for r in rows if r["finish_reason"] == "length"]
    lacos = [r["instance_id"] for r in rows if r["finish_reason"] == "length" or em_laco(r["response"])]
    print(f"CICLO {path}: acertos {acc}/{len(rows)} ({pc[0]}/{pc[1]}/{pc[2]}), ±10% {d10}, erro mediano {statistics.median(errs)*100 if errs else 0:.1f}%, "
          f"cortadas {len(cort)}, laços {len(lacos)} {lacos}, corretos {[r['instance_id'] for r in rows if r['correct']]}")
    json.dump(lacos, open(path.replace(".jsonl", ".laco.json"), "w"))
    with open(path.replace(".jsonl", ".laco.jsonl"), "w") as f:
        for r in rows:
            if r["instance_id"] in lacos: f.write(json.dumps(r) + "\n")
if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
