"""Pontos de decisão (sondas) extraídos das respostas limpas do Qwen3-14B.

Cada sonda é um prefixo (prompt de chat + começo da resposta do próprio
modelo, cortado logo antes do token decisivo) com o token correto e o token
errado. Tipos:
  os_tier : "For dimensions over 6" -> "2" (63-65 in, taxa $30) ou "5" (>65 in)
  ow_tier1: "For weights over "     -> "5" (<=70 lb) ou "7" (>70 lb)
  ow_tier2: "For weights over 5"    -> "0" (51-53 lb, $30) ou "3" (54-70 lb, $100)
  bag3    : "  - Third Bag: $"      -> "1" (150, EUA/Canadá) ou "2" (200)
Malas gratuitas de Business/First (limite de 70 lb) ficam fora das sondas de peso.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems, get_check_base_tables
from src.interp.experiment import chat_prompt

ITEM_RE = re.compile(r"Item (\d+) \(.*?\):\*?\*?\n", re.S)

def build(gen_path, tok):
    rows = [json.loads(l) for l in open(gen_path)]
    clean = [r for r in rows if r["scenario"] == "clean"]
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}
    tables = get_check_base_tables()
    probes = []
    for r in clean:
        c, idx = int(r["instance_id"][1]), int(r["instance_id"][3:])
        info = probs[c][idx]["info"]; bags = info["bag_list"][1:]
        base = [int(tables[min(3, i)][info["direction"]][info["customer_class"]][info["routine"]]) for i in range(len(bags))]
        resp = r["response"]
        base_prompt = chat_prompt(tok, r["question"], "one_shot")
        heads = list(ITEM_RE.finditer(resp))
        # taxa da 3a mala (só quando o modelo escreve a linha e o gabarito é 150 ou 200)
        m = re.search(r"- Third Bag: \$", resp)
        if m and len(base) >= 3 and base[2] in (150, 200):
            gold, wrong = ("1", "2") if base[2] == 150 else ("2", "1")
            probes.append(dict(instance=r["instance_id"], kind="bag3", item=None, gold=gold, wrong=wrong,
                               prefix=base_prompt + resp[:m.end()], info=f"{info['customer_class']}/{info['routine']}"))
        for j, h in enumerate(heads):
            n = int(h.group(1)); k = n - 2
            if k < 0 or k >= len(bags): continue
            end = heads[j + 1].start() if j + 1 < len(heads) else len(resp)
            body = resp[h.end():end]; off = h.end()
            b = bags[k]; sz = sum(b["size"]); w = b["weight"]
            comp = base[k] == 0 and info["customer_class"] in ("Business", "First")
            m = re.search(r"For dimensions over 6", body)
            if m and sz > 62:
                gold, wrong = ("2", "5") if sz <= 65 else ("5", "2")
                probes.append(dict(instance=r["instance_id"], kind="os_tier", item=n, gold=gold, wrong=wrong,
                                   prefix=base_prompt + resp[:off + m.end()], info=f"{sz} in"))
            m = re.search(r"For weights over ", body)
            if m and w > 50 and not comp:
                gold, wrong = ("5", "7") if w <= 70 else ("7", "5")
                probes.append(dict(instance=r["instance_id"], kind="ow_tier1", item=n, gold=gold, wrong=wrong,
                                   prefix=base_prompt + resp[:off + m.end()], info=f"{w} lb"))
                if w <= 70:
                    gold, wrong = ("0", "3") if w <= 53 else ("3", "0")
                    probes.append(dict(instance=r["instance_id"], kind="ow_tier2", item=n, gold=gold, wrong=wrong,
                                       prefix=base_prompt + resp[:off + m.end()] + "5", info=f"{w} lb"))
    return probes

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    probes = build(sys.argv[1], tok)
    import collections
    print(collections.Counter((p["kind"], p["gold"]) for p in probes))
    for p in probes:
        for t in (p["gold"], p["wrong"]):
            assert len(tok(t, add_special_tokens=False).input_ids) == 1, t
    out = Path(sys.argv[2]); out.write_text("\n".join(json.dumps(p) for p in probes))
    print(len(probes), "sondas ->", out)
