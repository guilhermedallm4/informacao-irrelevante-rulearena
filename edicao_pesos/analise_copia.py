"""Resume o diagnóstico de cópia (diag_copia.py): quais cabeças olham para o exemplo
1-shot nos pontos de decisão errados e empurram o token errado.

Pontuação de cópia da cabeça (L,H) = média, nas marcas erradas, de
  atenção_no_exemplo(L,H) × max(0, −DLA(L,H)).
Escreve um config com as k cabeças de maior pontuação zeradas em o_proj.
Uso: analise_copia.py <diag.pt> [k=8] [config_saida.json] [min_erros=5]
"""
import sys, json, torch, collections

def main(path, k=8, cfg_out=None, min_erros=5):
    D = torch.load(path); meta = D["meta"]; att = D["att"].float(); dla = D["dla"].float()
    ex = att[..., 2]                                   # [n, L, H] massa no exemplo
    marg = torch.tensor([m["margem"] for m in meta]); err = marg < 0
    por = collections.defaultdict(list)
    for i, m in enumerate(meta): por[(m["nome"], m["gold_txt"])].append(i)
    print(f"{len(meta)} marcas, {int(err.sum())} erradas; atenção média no exemplo (todas as cabeças, todas as marcas): {ex.mean():.3f}")
    print("\nPor classe com ≥%d erros: n, erros, margem média, DLA total das cabeças (erradas), atenção no exemplo (erradas vs certas, média das cabeças)" % min_erros)
    L, H = dla.shape[1:]; score = torch.zeros(L, H); n_err = 0
    for key, ix in sorted(por.items()):
        ix = torch.tensor(ix); e = ix[err[ix]]; c = ix[~err[ix]]
        if len(e) < min_erros: continue
        print(f"  {key[0]:12s} {key[1]!r:8s} n={len(ix):4d} erros={len(e):4d} margem={marg[ix].mean():+6.2f} DLA_cab(err)={dla[e].sum((1, 2)).mean():+6.2f} "
              f"att_ex err={ex[e].mean():.3f} cert={ex[c].mean() if len(c) else float('nan'):.3f}")
        score += (ex[e] * (-dla[e]).clamp(min=0)).sum(0); n_err += len(e)
    score /= max(n_err, 1)
    flat = score.flatten().argsort(descending=True)[:max(k, 20)]
    print(f"\nCabeças de maior pontuação de cópia (atenção no exemplo × empurrão ao token errado, marcas erradas):")
    print("  camada.cabeça  pontuação  att_ex(err)  DLA(err)  DLA(certas)  att_ex(certas)")
    e_all = err.nonzero().flatten(); c_all = (~err).nonzero().flatten()
    for f in flat:
        l, h = divmod(f.item(), H)
        print(f"  {l:2d}.{h:2d}        {score[l, h]:.4f}     {ex[e_all, l, h].mean():.3f}      {dla[e_all, l, h].mean():+.3f}    {dla[c_all, l, h].mean():+.3f}     {ex[c_all, l, h].mean():.3f}")
    tot_neg = (-dla[e_all]).clamp(min=0).sum((1, 2)).mean()
    top = flat[:k]; sel = [(divmod(f.item(), H)) for f in top]
    cobre = sum((-dla[e_all, l, h]).clamp(min=0).mean() for l, h in sel)
    print(f"\nAs {k} primeiras cobrem {cobre / tot_neg * 100:.1f}% do empurrão negativo total das cabeças nas marcas erradas "
          f"(empurrão negativo médio total {tot_neg:.2f}; margem média nas erradas {marg[e_all].mean():+.2f})")
    if cfg_out:
        json.dump({"escalas_cabecas": {f"{l}.{h}": 0.0 for l, h in sel}}, open(cfg_out, "w"), indent=1); print("config ->", cfg_out)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 8, a[3] if len(a) > 3 else None, int(a[4]) if len(a) > 4 else 5)
