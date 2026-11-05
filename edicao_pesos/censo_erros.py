"""Censo detalhado dos erros por item: cada decisão do modelo contra o gabarito."""
import json, sys, re, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems, get_check_base_tables, compute_oversize, compute_overweight
ITEM_RE = re.compile(r"Item (\d+) \(.*?\):\*?\*?\n", re.S)
def regiao(rt):
    if rt in ("U.S.", "Canada"): return "EUA/Canadá"
    if rt in ("China", "Japan", "South Korea", "Hong Kong", "India"): return "Ásia"
    if rt in ("Australia", "New Zealand"): return "Oceania"
    if rt in ("Europe", "Israel", "Qatar"): return "Europa"
    if rt in ("Panama", "South America", "Peru", "Colombia", "Ecuador"): return "Am. Sul"
    if rt == "Cuba": return "Cuba"
    return "Mex/Car/AmC"
def censo(path):
    rows = [json.loads(l) for l in open(path)]
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}; tables = get_check_base_tables()
    cnt = collections.Counter(); por_prob = collections.Counter(); itens = 0; itens_ok = 0; probs_sem_erro_item = 0
    detalhes = collections.defaultdict(collections.Counter); fora_padrao = set()
    for r in rows:
        c, idx = int(r["instance_id"][1]), int(r["instance_id"][3:]); info = probs[c][idx]["info"]
        bags = info["bag_list"][1:]; cls = info["customer_class"]; rt = info["routine"]; reg = regiao(rt)
        base = [int(tables[min(3, i)][info["direction"]][cls][rt]) for i in range(len(bags))]
        resp = r["response"]; heads = list(ITEM_RE.finditer(resp)); erros_prob = set()
        # lista "Checked Bag Fees"
        for m in re.finditer(r"- (First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth) Bag: \$(\d+)", resp):
            k = ["First","Second","Third","Fourth","Fifth","Sixth","Seventh","Eighth","Ninth","Tenth"].index(m.group(1))
            if k < len(base) and int(m.group(2)) != base[k]:
                e = f"lista: {m.group(1)} Bag ${m.group(2)} (gab ${base[k]})"; cnt[e.split(' $')[0] + f" [{reg}/{cls}]"] += 1; erros_prob.add("lista")
                detalhes["lista"][f"{m.group(1)} {reg} {cls}: ${m.group(2)} vs ${base[k]}"] += 1
        for j, h in enumerate(heads):
            k = int(h.group(1)) - 2
            if k < 0 or k >= len(bags): continue
            end = heads[j+1].start() if j+1 < len(heads) else len(resp); body = resp[h.end():end]
            b = bags[k]; sz = sum(b["size"]); w = b["weight"]; comp = base[k] == 0 and cls in ("Business", "First")
            g_os = compute_oversize(b, rt); g_ow = compute_overweight(b, rt, cls, comp); itens += 1; err = []
            mc = re.search(r"checking fee is \$(\d+)", body)
            if mc and int(mc.group(1)) != base[k]: err.append(f"checking fee mala {k+1}: ${mc.group(1)} vs ${base[k]} [{reg}/{cls}]")
            if sz > 62 and "Within the standard size" in body: err.append(f"tamanho {sz}: disse dentro do limite")
            if sz <= 62 and "Over the standard size" in body: err.append(f"tamanho {sz}: disse acima do limite")
            padrao = bool(re.search(r"Checking Fee:\n\s*- For the", body))
            if not padrao:
                fora_padrao.add(r["instance_id"]); continue
            mo = re.search(r"Oversize Fee:\n\s*- For dimensions over (\d+) inches up to (\d+) inches.*?\$(\d+)", body)
            if sz > 62:
                if not mo: 
                    if g_os and not re.search(r"Oversize Fee:.*?\$0|No oversize", body, re.S): err.append(f"oversize N: sem linha de taxa")
                else:
                    faixa_ok = (int(mo.group(1)) == 62) == (sz <= 65)
                    if not faixa_ok: err.append(f"oversize faixa {'63-65' if sz<=65 else '>65'}: disse over {mo.group(1)}")
                    elif int(mo.group(3)) != g_os: err.append(f"oversize valor {'63-65' if sz<=65 else '>65'} [{reg}]: ${mo.group(3)} vs ${g_os}")
            lim = 70 if comp else 50
            if w > lim and "Within the standard weight" in body: err.append(f"peso {w} (lim {lim}): disse dentro")
            if w <= lim and re.search(r"Over the standard weight limit", body) and not re.search(r"Within the standard weight", body): err.append(f"peso {w} (lim {lim}{' comp' if comp else ''}): disse acima")
            mw = re.search(r"Overweight Fee:\n\s*- For weights over (\d+) lbs up to (\d+) lbs.*?\$(\d+)", body)
            if g_ow > 0:
                if not mw: err.append(f"overweight {w} [{reg}/{cls}{' comp' if comp else ''}]: sem linha de taxa (gab ${g_ow})")
                else:
                    tier = "51-53" if w <= 53 else ("54-70" if w <= 70 else ">70")
                    faixa = {"50": "51-53", "53": "54-70", "70": ">70"}.get(mw.group(1), mw.group(1))
                    if faixa != tier: err.append(f"overweight faixa {tier}: disse over {mw.group(1)}")
                    elif int(mw.group(3)) != g_ow: err.append(f"overweight valor {tier} [{reg}/{cls}]: ${mw.group(3)} vs ${g_ow}")
            elif mw and int(mw.group(3)) > 0: err.append(f"overweight indevido {w} [{reg}/{cls}{' comp' if comp else ''}]: ${mw.group(3)} (gab $0)")
            mt = re.search(r"Total Fee for Item \d+: \$(\d+) \(checking\) \+ \$(\d+) .*?= \$([\d,]+)", body)
            if mt:
                chk, v, tot = int(mt.group(1)), int(mt.group(2)), int(mt.group(3).replace(",", ""))
                if tot != chk + v: err.append("soma do item errada")
                pos_os = int(mo.group(3)) if mo else 0; pos_ow = int(mw.group(3)) if mw else 0
                if v not in (max(pos_os, pos_ow),) and v != pos_os + pos_ow: err.append("valor do item != max(os,ow) declarado")
                if v == pos_os + pos_ow and pos_os and pos_ow: err.append("somou os+ow em vez do maior")
            if err:
                for e in err: cnt[re.sub(r"\d+ \(lim", "N (lim", re.sub(r"(tamanho|peso|overweight|overweight indevido) \d+", r"\1 N", e))] += 1; erros_prob.add(e.split(":")[0])
            else: itens_ok += 1
        if not erros_prob: probs_sem_erro_item += 1
        por_prob[len(erros_prob)] += 1
    print(f"{Path(path).stem}: itens {itens}, itens sem erro {itens_ok} ({itens_ok/itens:.0%}); problemas sem erro de item: {probs_sem_erro_item}/90; problemas com itens fora do formato padrão: {len(fora_padrao)}; nº de tipos de erro por problema: {sorted(por_prob.items())}")
    for k, v in cnt.most_common(40): print(f"  {v:3d}  {k}")
    for k, d in detalhes.items(): print("  detalhe", k, d.most_common(12))
if __name__ == "__main__":
    for p in sys.argv[1:]: censo(p)
