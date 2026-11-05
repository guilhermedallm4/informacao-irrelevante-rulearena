"""Sonda linear fechada: o 1º item da ordem do gabarito é legível no fluxo residual na
posição em que o modelo escreve "Items to be checked: Items ▸"? Ridge um-contra-todos
com validação leave-one-out, por camada, comparado com a maioria (ordem natural, item 2)
e medido à parte nos problemas em que a ordem muda. Uso: sonda_ordem.py <res_tracos_reord.pt>"""
import sys, torch, collections
D = torch.load(sys.argv[1]); meta = D["meta"]; acts = D["acts"].float(); cam = D["camadas"]
ix = {}; 
for i, m in enumerate(meta):
    if m["nome"] == "ordem_lista" and m["instance"] not in ix: ix[m["instance"]] = i
ids = sorted(ix); I = torch.tensor([ix[k] for k in ids]); y = [meta[i]["gold_txt"] for i in I.tolist()]
hard = torch.tensor([meta[i]["gold_txt"] != "2" for i in I.tolist()])
cls = sorted(set(y)); Y = torch.tensor([[float(v == c) for c in cls] for v in y]); n = len(ids)
print(f"{n} problemas, {int(hard.sum())} com 1º item ≠ 2; maioria (sempre 2) acerta {n - int(hard.sum())}/{n}")
for ci, L in enumerate(cam):
    X = acts[I, ci]; X = X - X.mean(0)
    for lam_rel in (1.0, 10.0):
        acc = 0; acc_h = 0
        K = X @ X.T; lam = lam_rel * K.diag().mean()
        for t in range(n):                       # leave-one-out, forma dual
            tr = torch.arange(n) != t
            Ym = Y[tr].mean(0); A = K[tr][:, tr] + lam * torch.eye(n - 1); a = torch.linalg.solve(A, Y[tr] - Ym)
            pred = (K[t, tr] @ a + Ym).argmax().item(); ok = cls[pred] == y[t]; acc += ok; acc_h += ok and hard[t].item()
        print(f"camada {L:2d} λ_rel={lam_rel:4.0f}: LOO acerto {acc}/{n}, nos {int(hard.sum())} difíceis {acc_h}")
