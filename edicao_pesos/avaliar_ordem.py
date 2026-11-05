"""Avalia a reordenação das malas (ordem do gabarito oficial).

Modo tracos <res.pt>: saída de medir_traco.py/coletar_residual.py em traços reordenados;
  conta, por problema, se todas as marcas ordem_lista (e ordem_item) são top-1,
  separando os problemas cuja ordem do gabarito difere da do enunciado.
Modo livre <respostas.jsonl>: lê a ordem em que o modelo enumerou os itens
  ("Items to be checked: Items ..." e "k. Item n") e compara com a ordem do gabarito.
Uso: avaliar_ordem.py tracos <res.pt> | livre <respostas.jsonl>
"""
import sys, json, re, collections, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.airline.gold import load_airline_problems
from src.interp.experiment import sample_problems
from edicao_pesos.traco_gabarito import ordem_gabarito

def problemas():
    probs = {c: load_airline_problems(c) for c in (0, 1, 2)}; out = {}
    for s in sample_problems(0.3, 42):
        p = probs[s["complexity"]][s["problem_idx"]]; info = p["info"]; bags = info["bag_list"][1:]
        ordem = ordem_gabarito(bags, info["routine"], info["customer_class"])
        out[s["instance_id"]] = dict(gold=[i + 2 for i in ordem], natural=list(range(2, len(bags) + 2)), muda=ordem != list(range(len(bags))), cls=info["customer_class"])
    return out

def tracos(path):
    P = problemas(); meta = torch.load(path)["meta"]
    ok = collections.defaultdict(lambda: collections.defaultdict(lambda: True)); primeiro = {}
    for m in meta:
        if m["nome"] in ("ordem_lista", "ordem_item"):
            certo = m["top"][0] == m["gold"]; ok[m["instance"]][m["nome"]] &= certo
            if m["nome"] == "ordem_lista" and m["instance"] not in primeiro: primeiro[m["instance"]] = certo
    for grupo, sel in (("ordem diferente do enunciado", True), ("mesma ordem", False)):
        ids = [i for i, p in P.items() if p["muda"] == sel]
        print(f"{grupo} ({len(ids)} problemas): lista inteira certa {sum(ok[i]['ordem_lista'] for i in ids)}, "
              f"1º item da lista certo {sum(primeiro.get(i, False) for i in ids)}, enumeração inteira certa {sum(ok[i]['ordem_item'] for i in ids)}")

def livre(path):
    P = problemas(); rows = [json.loads(l) for l in open(path)]
    res = collections.defaultdict(list)
    for r in rows:
        p = P[r["instance_id"]]; t = r["response"]
        m = re.search(r"to be checked:?\**\s*\(?Items?\s*([\d,\s and]+)", t)
        lista = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
        enum = [int(x) for x in re.findall(r"(?m)^[ \t#]*\**\d+\.\s*\**\s*Item (\d+)", t)]
        res[p["muda"]].append(dict(id=r["instance_id"], lista=lista, enum=enum, gold=p["gold"], natural=p["natural"], correct=r["correct"]))
    for muda in (True, False):
        rs = res[muda]; n = len(rs)
        print(f"{'ordem diferente do enunciado' if muda else 'mesma ordem'} ({n} problemas): lista = gabarito {sum(r['lista'] == r['gold'] for r in rs)}, "
              f"lista = enunciado {sum(r['lista'] == r['natural'] for r in rs)}, enumeração = gabarito {sum(r['enum'] == r['gold'] for r in rs)}, "
              f"enumeração = enunciado {sum(r['enum'] == r['natural'] for r in rs)}, outra/ilegível {sum(r['enum'] not in (r['gold'], r['natural']) for r in rs)}, acertos {sum(r['correct'] for r in rs)}")
        if muda:
            for r in rs: print(f"   {r['id']}: gabarito {r['gold']} | lista {r['lista']} | enumeração {r['enum']}{'  ✔' if r['correct'] else ''}")

if __name__ == "__main__":
    (tracos if sys.argv[1] == "tracos" else livre)(sys.argv[2])
