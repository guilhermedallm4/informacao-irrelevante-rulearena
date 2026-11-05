"""Gera um config novo a partir de outro, ajustando α por classe à mão.
Uso: ajustar_config.py <cfg_base.json> <cfg_saida.json> "nome:'tok'=α" ...  [--v4 a23,a24,a31 | --sem-v4]
Ex.: ajustar_config.py edicao_pesos/config_c2.json edicao_pesos/config_c3.json "peso_lim:' Within'=200" "os_fee:'2'=450" --v4 500,500,300"""
import sys, json, re
D = "results/edicao_pesos/deltas/"
cfg = json.load(open(sys.argv[1])); out = sys.argv[2]; args = sys.argv[3:]
nomes = {}
for e in cfg.get("edicao_up", []):
    nomes[e["camada"]] = __import__("torch").load(e["arquivo"])["classes"]
pular = {args[args.index("--v4") + 1]} if "--v4" in args else set()
for a in list(args):
    if a.startswith("--") or a in pular: continue
    k, v = a.rsplit("=", 1); k = k.strip(); achou = False
    for e in cfg.get("edicao_up", []):
        for i, n in enumerate(nomes[e["camada"]]):
            if n == k: e["fatores"][i] = float(v); achou = True
    assert achou, f"classe {k!r} não encontrada em {[n for v in nomes.values() for n in v]}"
if "--sem-v4" in args: cfg.pop("edicao_fatorada", None)
if "--v4" in args:
    a = [float(x) for x in args[args.index("--v4") + 1].split(",")]
    cfg["edicao_fatorada"] = [{"camada": L, "arquivo": f"{D}fatores_v4_L{L}.pt", "fator": x} for L, x in zip((23, 24, 31), a)]
json.dump(cfg, open(out, "w"), indent=1)
print(out, {n: e["fatores"][i] for e in cfg.get("edicao_up", []) for i, n in enumerate(nomes[e["camada"]])}, cfg.get("edicao_fatorada", "sem v4"))
