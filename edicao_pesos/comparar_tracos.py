"""Compara o acerto top-1 por classe (decisão, token correto) entre duas medições
por teacher forcing (saídas de medir_traco.py, coletar_residual.py ou diag_copia.py).
Uso: comparar_tracos.py <a.pt> <b.pt> [min_n=10]"""
import sys, torch, collections

def acertos(path):
    meta = torch.load(path)["meta"]; c = collections.defaultdict(lambda: [0, 0])
    for m in meta:
        ok = m["top1_ok"] if "top1_ok" in m else (m["top"][0] == m["gold"])
        k = (m["nome"], m["gold_txt"]); c[k][1] += 1; c[k][0] += ok
    return c

def main(a, b, min_n=10):
    A, B = acertos(a), acertos(b); ta = tb = n = 0
    print(f"{'classe':26s} {'n':>5s} {'A':>5s} {'B':>5s} {'Δ':>5s}")
    for k in sorted(set(A) | set(B)):
        na, nb = A.get(k, [0, 0]), B.get(k, [0, 0])
        if na[1] < min_n: continue
        ta += na[0]; tb += nb[0]; n += na[1]
        flag = " <<" if abs(nb[0] - na[0]) >= max(3, 0.05 * na[1]) else ""
        print(f"{k[0] + ' ' + repr(k[1]):26s} {na[1]:5d} {na[0]:5d} {nb[0]:5d} {nb[0] - na[0]:+5d}{flag}")
    print(f"total (classes com n ≥ {min_n}): A {ta}/{n} = {ta / n * 100:.1f}%, B {tb}/{n} = {tb / n * 100:.1f}%")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
