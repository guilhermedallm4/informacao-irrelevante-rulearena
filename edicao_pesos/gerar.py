"""Gera as respostas limpas dos 90 problemas com o modelo (editado ou não).

Uso: python edicao_pesos/gerar.py <saida.jsonl> [config.json|-] [gpu] [lote]
Mesmo protocolo do estudo: 1-shot, decodificação gulosa, 4096 tokens, lote 16.
"""
import json, sys, torch, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.interp.experiment import sample_problems, chat_prompt, generate_greedy, _evaluate
from edicao_pesos.edicoes import aplicar

MODEL = "Qwen/Qwen3-14B"

def main(out, cfg_path, gpu, lote):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=f"cuda:{gpu}",
                                                 attn_implementation="sdpa").eval()
    edicoes = aplicar(model, tok, json.load(open(cfg_path))) if cfg_path else []
    for l in edicoes: print("  edição:", l)
    sample = sample_problems(0.3, 42)
    prompts = [chat_prompt(tok, s["question"], "one_shot") for s in sample]
    outs = generate_greedy(model, tok, prompts, 4096, lote, desc="clean")
    rows = []
    for s, o in zip(sample, outs):
        row = dict(instance_id=s["instance_id"], complexity=s["complexity"], gold=s["gold"], scenario="clean",
                   variant="clean", distractor=None, question=s["question"], key=f"{s['instance_id']}|clean",
                   response=o["text"], finish_reason=o["finish_reason"], n_tokens=o["n_tokens"])
        row.update(_evaluate(row["response"], o["finish_reason"], s["gold"]))
        rows.append(row)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in rows)
    por_c = collections.Counter(r["complexity"] for r in rows if r["correct"])
    dentro10 = sum(1 for r in rows if r["pred_cents"] and abs(r["pred_cents"]/100 - r["gold"]) <= 0.1*r["gold"])
    sem = sum(1 for r in rows if r["pred_cents"] is None)
    print(f"RESULTADO {out}: acertos {acc}/90 (C0/C1/C2 = {por_c[0]}/{por_c[1]}/{por_c[2]}), a ±10%: {dentro10}, sem resposta: {sem}, edições: {edicoes}")

if __name__ == "__main__":
    a = sys.argv
    main(a[1], None if len(a) < 3 or a[2] == "-" else a[2], int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 16)
