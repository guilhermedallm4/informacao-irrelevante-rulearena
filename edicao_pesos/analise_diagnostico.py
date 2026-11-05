"""Resume o diagnóstico: acurácia das sondas e cabeças que mais empurram para o token errado."""
import sys, torch, collections
recs = torch.load(sys.argv[1])
by = collections.defaultdict(list)
for r in recs: by[(r["kind"], r["gold"])].append(r)
print("== Acurácia token a token (sem edição)")
for k in sorted(by):
    rs = by[k]
    print(f"{k[0]:9s} gab={k[1]} n={len(rs):3d} correto>errado={sum(r['logit_diff']>0 for r in rs)/len(rs):5.1%} "
          f"top1=gab={sum(r['top1']==r['gold'] for r in rs)/len(rs):5.1%} dif.média={sum(r['logit_diff'] for r in rs)/len(rs):+.2f}")
L, H = recs[0]["dla"].shape
nomes = ["sistema", "regras", "exemplo", "pergunta", "resposta"]
def top_heads(rs, n=12, sinal=-1):
    dla = torch.stack([r["dla"] for r in rs]).mean(0)
    att = torch.stack([r["att"] for r in rs]).mean(0)
    flat = (sinal * dla).flatten().topk(n)
    out = []
    for v, i in zip(flat.values, flat.indices):
        l, h = divmod(i.item(), H)
        out.append((l, h, dla[l, h].item(), att[l, h].tolist()))
    return out, dla
for grupo, filtro in [("erros 63-65in (os_tier gab=2)", lambda r: r["kind"]=="os_tier" and r["gold"]=="2"),
                      ("erros 51-53lb (ow_tier2 gab=0)", lambda r: r["kind"]=="ow_tier2" and r["gold"]=="0"),
                      ("erros 54-70lb (ow_tier2 gab=3)", lambda r: r["kind"]=="ow_tier2" and r["gold"]=="3"),
                      ("<=70lb (ow_tier1 gab=5)", lambda r: r["kind"]=="ow_tier1" and r["gold"]=="5"),
                      ("3a mala $150 (bag3 gab=1)", lambda r: r["kind"]=="bag3" and r["gold"]=="1")]:
    rs = [r for r in recs if filtro(r)]
    print(f"\n== {grupo}: n={len(rs)}  (DLA médio por cabeça na direção correto-errado; negativo = empurra para o erro)")
    heads, dla = top_heads(rs)
    print("   soma DLA atenção =", f"{dla.sum().item():+.2f}", " soma por camada (10 piores):",
          [(i, round(v, 2)) for i, v in sorted(enumerate(dla.sum(1).tolist()), key=lambda x: x[1])[:10]])
    for l, h, v, a in heads:
        print(f"   L{l:2d}.H{h:2d}  DLA={v:+.3f}  atenção: " + " ".join(f"{n}={x:.2f}" for n, x in zip(nomes, a)))
    heads_pos, _ = top_heads(rs, n=6, sinal=+1)
    print("   cabeças que mais ajudam:", ", ".join(f"L{l}.H{h} ({v:+.2f}, ex={a[2]:.2f} reg={a[1]:.2f})" for l, h, v, a in heads_pos))
