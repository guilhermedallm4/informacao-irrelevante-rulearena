"""Traço-gabarito: a resposta ideal de cada problema no formato do exemplo 1-shot,
com cada ponto de decisão marcado (offset do caractere onde começa o token
decisivo, nome da decisão, texto correto que começa ali).

Decisões marcadas (nome: o que o modelo tem de escolher):
  lista_bagN   : valor na lista "Checked Bag Fees" (1º dígito)
  tam_lim      : "Over"/"Within" the standard size limit
  peso_lim     : "Over"/"Within" the standard weight limit
  peso_faixa   : "but under 53" / "but under 70" / "and over 70"  (token após "50 lbs ")
  chk_fee      : "the checking fee is $" ▸ 1º dígito
  os_faixa     : "For dimensions over " ▸ "62" ou "65" (2º dígito)
  os_fee       : oversize "the fee is $" ▸ 1º dígito
  ow_faixa     : "For weights over " ▸ "50"/"53"/"70" (1º dígito e 2º dígito)
  ow_fee       : overweight "the fee is $" ▸ 1º dígito
  item_valor   : "Total Fee for Item N: $chk (checking) + $" ▸ 1º dígito (o maior entre os/ow)
  item_total   : "= $" ▸ 1º dígito do total do item
  total        : "The total cost is $" ▸ 1º dígito
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems, get_check_base_tables, compute_oversize, compute_overweight
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
ORDC = [o.capitalize() for o in ORD]
ASIA = ("India", "China", "Japan", "South Korea", "Hong Kong")

def regiao_frase(rt):
    if rt == "U.S.": return "within the U.S."
    return f"between the U.S. and {rt}"

def traco(p):
    """Devolve (texto, marcas) para um problema do RuleArena (dict com prompt e info)."""
    info = p["info"]; q = p["prompt"]; tables = get_check_base_tables()
    nome = q.split(" is a ")[0]
    cls = info["customer_class"]; rt = info["routine"]; direc = info["direction"]
    bags_all = info["bag_list"]; carry = bags_all[0]; bags = bags_all[1:]
    base = [int(tables[min(3, i)][direc][cls][rt]) for i in range(len(bags))]
    out = []; marcas = []
    def w(s): out.append(s)
    def pos(): return sum(len(s) for s in out)
    def marca(nome_, texto):   # marca no ponto atual e escreve o texto; números ganham marca do 2º dígito
        marcas.append((pos(), nome_, texto))
        if texto[0].isdigit() and len(texto) > 1 and texto[1].isdigit(): marcas.append((pos() + 1, nome_ + "_d2", texto[1:]))
        w(texto)
    w(f"To calculate the total cost for {nome}, we need to consider the flight ticket cost, checked bag fees, and any overweight or oversize fees according to American Airlines' policies.\n\n")
    w(f"Flight Ticket Cost:\n- {nome}'s flight ticket: ${info['base_price']}\n\n")
    w(f"Carry-on Allowance:\n- {nome} is allowed 1 carry-on bag and 1 personal item.\n")
    cd = carry["size"]
    w(f"- Item 1 ({carry['name'].capitalize()}):\n  - Dimensions: {cd[0]} x {cd[1]} x {cd[2]} inches\n  - Weight: {carry['weight']} lbs\n  - This fits within the carry-on size limit of 22 x 14 x 9 inches.\n  - Fee: $0 (Allowed as carry-on bag)\n")
    w(f"- {nome} does not have any items that fit the personal item dimensions (18 x 14 x 8 inches). So, no personal item is carried.\n\n")
    n = len(bags); itens = ", ".join(str(i + 2) for i in range(n - 1)) + f", and {n + 1}"
    w(f"Checked Bags:\n- Items to be checked: Items {itens}\n")
    livres = sum(1 for b in base if b == 0)
    if cls == "Main Plus": w(f"- {nome} is a Main Plus passenger, which includes 1 extra free checked bag in addition to the Main Cabin allowance, for a total of {livres} free checked bag{'s' if livres != 1 else ''}.\n")
    else: w(f"- {nome} is a {cls} passenger, which includes {livres} free checked bag{'s' if livres != 1 else ''}.\n")
    w("- Checked Bag Fees:\n")
    for i, b in enumerate(base):
        w(f"  - {ORDC[i]} Bag: $"); marca(f"lista_bag{min(i+1,4)}", f"{b}{' (free)' if b == 0 else ''}\n")
    w("\nFees for Each Checked Bag:\n\n")
    tot_bags = 0; fees = []
    for i, b in enumerate(bags):
        sz = sum(b["size"]); wt = b["weight"]; comp = base[i] == 0 and cls in ("Business", "First")
        lim = 70 if comp else 50
        g_os = compute_oversize(b, rt); g_ow = compute_overweight(b, rt, cls, comp)
        w(f"{i+1}. Item {i+2} ({b['name'].capitalize()}):\n   - Dimensions: {b['size'][0]} x {b['size'][1]} x {b['size'][2]} inches\n     - Total dimensions: {b['size'][0]} + {b['size'][1]} + {b['size'][2]} = {sz} inches\n     -")
        if sz > 62: marca("tam_lim", " Over the standard size limit of 62 inches.\n")
        else: marca("tam_lim", " Within the standard size limit of 62 inches.\n")
        w(f"   - Weight: {wt} lbs\n     -")
        if wt <= lim: marca("peso_lim", f" Within the standard weight limit of {lim} lbs.\n")
        else:
            marca("peso_lim", f" Over the standard weight limit of {lim} lbs")
            if wt <= 53: marca("peso_faixa", " but under 53 lbs.\n")
            elif wt <= 70: marca("peso_faixa", " but under 70 lbs.\n")
            else: marca("peso_faixa", " and over 70 lbs but under 100 lbs.\n")
        w(f"   - Checking Fee:\n     - For the {ORD[i]} checked bag, the checking fee is $"); marca("chk_fee", f"{base[i]}.\n")
        if g_os:
            w("   - Oversize Fee:\n     - For dimensions over 6")
            if sz <= 65: marca("os_faixa", "2 inches up to 65 inches"); w(", the fee is $"); marca("os_fee", f"{g_os}.\n")
            else: marca("os_faixa", "5 inches up to 115 inches"); w(f" {regiao_frase(rt)}, the fee is $"); marca("os_fee", f"{g_os}.\n")
        if g_ow:
            w("   - Overweight Fee:\n     - For weights over ")
            if wt <= 53: marca("ow_faixa", "50 lbs up to 53 lbs")
            elif wt <= 70: marca("ow_faixa", "53 lbs up to 70 lbs")
            else: marca("ow_faixa", "70 lbs up to 100 lbs")
            w(", the fee is $"); marca("ow_fee", f"{g_ow}.\n")
        if g_os and g_ow: w("   - The higher of oversize and overweight fee should apply.\n")
        v = max(g_os, g_ow); tot = base[i] + v; fees.append(tot); tot_bags += tot
        w(f"   - Total Fee for Item {i+2}: ${base[i]} (checking) + $")
        if v == 0: marca("item_valor", "0 (No overweight or oversize fees)")
        else: marca("item_valor", f"{v} ({'oversize' if g_os >= g_ow else 'overweight'})")
        w(" = $"); marca("item_total", f"{tot}\n")
    w("Summary of Baggage Fees:\n")
    for i, f in enumerate(fees): w(f"  - Item {i+2}: ${f}\n")
    w("Total Baggage Fees: " + " + ".join(f"${f} (Item {i+2})" for i, f in enumerate(fees)) + f" = ${tot_bags}\n")
    total = info["base_price"] + tot_bags
    w(f"Total Cost:\n- Flight Ticket: ${info['base_price']}\n- Total Baggage Fees: ${tot_bags}\n- The total cost is $"); marca("total", f"{total:,}.")
    return "".join(out), marcas, total

if __name__ == "__main__":
    from src.interp.experiment import sample_problems
    from src.airline.gold import compute_airline_gold
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}
    sample = sample_problems(0.3, 42)
    for s in sample[:1]:
        p = probs[s["complexity"]][s["problem_idx"]]
        t, marcas, total = traco(p); print(t); print(marcas[:12]); print("gold", s["gold"], total)
    ok = 0
    for s in sample:
        p = probs[s["complexity"]][s["problem_idx"]]; t, marcas, total = traco(p); ok += (total == s["gold"])
    print("traços com total igual ao gabarito:", ok, "/", len(sample))
