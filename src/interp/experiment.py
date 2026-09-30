"""Irrelevant-information experiment on RuleArena airline, fully in HF transformers.

For each sampled problem:

  1. Generation (greedy, HF `generate`): the model answers
       - clean:            the original question;
       - control:          the original question again, in a different batch
                           (noise floor of batched greedy decoding);
       - prompt:<key>:     the question with a distractor sentence inserted
                           in the middle of the item list;
       - reasoning:<key>:  the clean reasoning cut mid-calculation, the
                           distractor appended, and the model continues;
       - reasoning:control: the same cut without distractor.
  2. Capture (teacher-forced forward pass with hooks): the clean reasoning is
     replayed under each condition and activations are recorded at the
     positions that predict the clean final answer. Every condition therefore
     reads out the *same* answer tokens, and differences come only from the
     distractor. A "clean_padded" condition replays the clean text with
     masked left padding: its difference to "clean" is the numerical-noise
     floor for all activation metrics.

Outputs, under results/irrelevant_info/<run_id>/:
  manifest.json, generations.jsonl, captures/<instance_id>.pt
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ..airline.gold import compute_airline_gold, load_airline_problems
from ..airline.prompt_builder import build_airline_messages
from ..evaluator_v2_1 import evaluate_response
from . import activations as A
from .distractors import (find_reasoning_cut, insert_in_question,
                          insert_in_reasoning, split_final_answer)


# ── Sampling ─────────────────────────────────────────────────────────────

def sample_problems(fraction: float, seed: int, complexities=(0, 1, 2)) -> list[dict]:
    """Stratified random sample: `fraction` of each complexity level."""
    rng = random.Random(seed)
    sample = []
    for c in complexities:
        problems = load_airline_problems(c)
        k = round(fraction * len(problems))
        for idx in sorted(rng.sample(range(len(problems)), k)):
            p = problems[idx]
            sample.append({"instance_id": f"c{c}_{idx:03d}", "complexity": c,
                           "problem_idx": idx, "question": p["prompt"],
                           "gold": int(compute_airline_gold(p["info"]))})
    return sample


# ── Generation ───────────────────────────────────────────────────────────

def chat_prompt(tok, question: str, prompt_mode: str, enable_thinking: bool = False) -> str:
    messages = build_airline_messages(question, prompt_mode=prompt_mode)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)


# Plain greedy decoding. `generate` starts from the model's own
# generation_config.json, which may carry more than sampling settings:
# Qwen2.5 ships repetition_penalty=1.05 and LFM2.5-2.6B 1.1. Every setting
# that changes the greedy choice is reset here, so all models decode alike.
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
              repetition_penalty=1.0, no_repeat_ngram_size=0)


@torch.no_grad()
def generate_greedy(model, tok, prompts: list[str], max_new_tokens: int,
                    batch_size: int, desc: str = "generate", on_batch=None) -> list[dict]:
    """Greedy decoding in left-padded batches. Returns text + finish reason.

    `on_batch(start_index, results)` is called after every batch, so callers
    can persist progress. A batch that runs out of GPU memory is split in
    half and retried.
    """
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def attempt(chunk):
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(A.input_device(model))
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **GREEDY)
        return out[:, enc.input_ids.shape[1]:].tolist()

    def run(chunk):
        # On CUDA OOM, split the batch in half and retry each part. The retry
        # happens *after* the except block: inside it, the exception's
        # traceback keeps the failed attempt's tensors alive.
        oom = False
        try:
            seqs = attempt(chunk)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            oom = True
        gc.collect()
        torch.cuda.empty_cache()
        if oom:
            half = len(chunk) // 2
            print(f"  OOM com lote de {len(chunk)}; dividindo em {half} + {len(chunk) - half}")
            return run(chunk[:half]) + run(chunk[half:])
        batch = []
        for seq in seqs:
            stop = next((j for j, t in enumerate(seq) if t in eos), None)
            new = seq if stop is None else seq[:stop]
            batch.append({"text": tok.decode(new, skip_special_tokens=True),
                          "finish_reason": "length" if stop is None else "stop",
                          "n_tokens": len(new)})
        return batch

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = run(prompts[i:i + batch_size])
        results.extend(batch)
        if on_batch is not None:
            on_batch(i, batch)
    return results


def _evaluate(response: str, finish_reason: str, gold: int) -> dict:
    ev = evaluate_response(response, finish_reason, starts_inside_reasoning=False)
    cents = ev["value_cents"] if ev["strict_evaluable"] else None
    return {"pred_cents": cents, "parse_status": ev["parse_status"],
            "correct": cents is not None and cents == gold * 100}


def _run_phase(model, tok, jobs: list[dict], done: dict, partial_path: Path,
               max_new_tokens: int, batch_size: int, desc: str) -> list[dict]:
    """Generate the jobs not yet in `done`, appending each batch to `partial_path`.

    A job is {"key", "prompt", "row"}: `row` holds the metadata and receives
    the generation. Returns the finished rows in job order.
    """
    pending = [j for j in jobs if j["key"] not in done]

    def save(start, outs):
        with open(partial_path, "a") as f:
            for job, out in zip(pending[start:start + len(outs)], outs):
                row = dict(job["row"], key=job["key"])
                row["response"] = row.pop("forced_prefix_for_response", "") + out["text"]
                row.update(finish_reason=out["finish_reason"], n_tokens=out["n_tokens"])
                row.update(_evaluate(row["response"], out["finish_reason"], row["gold"]))
                done[job["key"]] = row
                f.write(json.dumps(row) + "\n")

    if pending:
        generate_greedy(model, tok, [j["prompt"] for j in pending], max_new_tokens,
                        batch_size, desc, on_batch=save)
    return [done[j["key"]] for j in jobs]


def _base_row(s: dict) -> dict:
    return {"instance_id": s["instance_id"], "complexity": s["complexity"], "gold": s["gold"]}


def run_generation(model, tok, sample: list[dict], distractors: dict[str, str],
                   out_path: Path, prompt_mode: str = "one_shot",
                   max_new_tokens: int = 4096, batch_size: int = 8,
                   seed: int = 42) -> list[dict]:
    """Generate every condition and write one JSON line per (instance, condition).

    Progress is appended batch by batch to `<out_path>.partial`; rerunning
    after an interruption skips what is already there. (Batches formed on
    resume can differ from an uninterrupted run, which is within the noise
    that the control condition measures.)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    done = {}
    if partial.exists():
        for line in open(partial):
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
        print(f"Retomando: {len(done)} gerações já feitas em {partial.name}")

    # Clean + prompt distractors: independent of the model's output
    jobs = []
    for s in sample:
        jobs.append({"key": f"{s['instance_id']}|clean", "prompt": chat_prompt(tok, s["question"], prompt_mode),
                     "row": dict(_base_row(s), scenario="clean", variant="clean",
                                 distractor=None, question=s["question"])})
        for key, sentence in distractors.items():
            q = insert_in_question(s["question"], sentence)
            jobs.append({"key": f"{s['instance_id']}|prompt:{key}", "prompt": chat_prompt(tok, q, prompt_mode),
                         "row": dict(_base_row(s), scenario="prompt", variant=key,
                                     distractor=sentence, question=q)})
    rows = _run_phase(model, tok, jobs, done, partial, max_new_tokens, batch_size, "clean + prompt")
    clean = {r["instance_id"]: r for r in rows if r["scenario"] == "clean"}

    # Control: the clean prompts again, shuffled so batch composition differs
    order = list(range(len(sample)))
    random.Random(seed + 1).shuffle(order)
    jobs = [{"key": f"{s['instance_id']}|control", "prompt": chat_prompt(tok, s["question"], prompt_mode),
             "row": dict(_base_row(s), scenario="control", variant="control",
                         distractor=None, question=s["question"])}
            for s in (sample[i] for i in order)]
    rows += _run_phase(model, tok, jobs, done, partial, max_new_tokens, batch_size, "control")

    # Reasoning distractors: cut the clean reasoning mid-calculation and continue
    jobs = []
    for s in sample:
        c = clean[s["instance_id"]]
        if c["finish_reason"] != "stop" or split_final_answer(c["response"]) is None:
            continue
        cut, kind = find_reasoning_cut(c["response"])
        base = chat_prompt(tok, s["question"], prompt_mode)
        prefix = c["response"][:cut]
        for key, sentence in [("control", None)] + list(distractors.items()):
            forced = prefix if sentence is None else insert_in_reasoning(prefix, sentence)
            jobs.append({"key": f"{s['instance_id']}|reasoning:{key}", "prompt": base + forced,
                         "row": dict(_base_row(s), scenario="reasoning", variant=key,
                                     distractor=sentence, question=s["question"],
                                     cut_char=cut, cut_kind=kind, forced_prefix=forced,
                                     forced_prefix_for_response=forced)})
    rows += _run_phase(model, tok, jobs, done, partial, max_new_tokens, batch_size, "reasoning")

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


# ── Capture ──────────────────────────────────────────────────────────────

NOISE_LEFT_PAD = 8   # padding tokens for the numerical-noise reference condition

def _summarize(model, tok, cap: A.Capture, top_k: int = 3) -> dict:
    lens = A.logit_lens(model, cap, top_k=top_k)
    dla = A.direct_logit_attribution(model, cap)
    return {
        "target_logp": lens["target_logp"], "target_rank": lens["target_rank"],
        "top_ids": lens["top_ids"], "top_logp": lens["top_logp"],
        "dla_attn": dla["attn"], "dla_mlp": dla["mlp"], "dla_embed": dla["embed"],
        "convergence_layer": A.convergence_layer(lens["target_rank"]),
        # neurons and residual at the first answer position, fp16 to save disk
        "mlp_act_pos0": cap.mlp_act[:, 0].half().cpu(),
        "resid_pos0": cap.resid[:, 0].half().cpu(),
        "n_prefix_tokens": cap.n_prefix_tokens,
    }


def teacher_forced_prefixes(tok, clean_row: dict, distractors: dict[str, str],
                            reasoning_rows: dict, prompt_mode: str) -> dict[str, str]:
    """Prefix text (up to "$") for every condition, all replaying the clean reasoning."""
    upto_dollar, _ = split_final_answer(clean_row["response"])
    q = clean_row["question"]
    prefixes = {"clean": chat_prompt(tok, q, prompt_mode) + upto_dollar}
    for key, sentence in distractors.items():
        prefixes[f"prompt:{key}"] = (chat_prompt(tok, insert_in_question(q, sentence), prompt_mode)
                                     + upto_dollar)
        rrow = reasoning_rows.get(key)
        if rrow is not None:
            cut = rrow["cut_char"]
            prefixes[f"reasoning:{key}"] = (chat_prompt(tok, q, prompt_mode)
                                            + rrow["forced_prefix"] + upto_dollar[cut:])
    return prefixes


def run_capture(model, tok, generations: list[dict], distractors: dict[str, str],
                out_dir: Path, prompt_mode: str = "one_shot") -> None:
    """Teacher-forced capture for every instance whose clean answer parses."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_inst: dict[str, dict] = {}
    for r in generations:
        by_inst.setdefault(r["instance_id"], {"reasoning": {}})
        if r["scenario"] == "clean":
            by_inst[r["instance_id"]]["clean"] = r
        elif r["scenario"] == "reasoning" and r["variant"] != "control":
            by_inst[r["instance_id"]]["reasoning"][r["variant"]] = r

    for inst_id, d in tqdm(by_inst.items(), desc="capture"):
        path = out_dir / f"{inst_id}.pt"
        clean = d.get("clean")
        if path.exists() or clean is None or clean["finish_reason"] != "stop":
            continue
        split = split_final_answer(clean["response"])
        if split is None:
            continue
        answer = split[1]
        prefixes = teacher_forced_prefixes(tok, clean, distractors, d["reasoning"], prompt_mode)

        clean_cap = A.capture(model, tok, prefixes["clean"], answer)
        record = {"instance_id": inst_id, "answer": answer,
                  "answer_tokens": [tok.decode([t]) for t in clean_cap.target_ids.tolist()],
                  "conditions": {"clean": _summarize(model, tok, clean_cap)}, "compare": {}}
        # Numerical-noise floor: same clean text, only the tensor shape changes
        padded = A.capture(model, tok, prefixes["clean"], answer, left_pad=NOISE_LEFT_PAD)
        record["conditions"]["clean_padded"] = _summarize(model, tok, padded)
        record["compare"]["clean_padded"] = A.compare(clean_cap, padded)
        del padded
        for cond, prefix in prefixes.items():
            if cond == "clean":
                continue
            cap = A.capture(model, tok, prefix, answer)
            record["conditions"][cond] = _summarize(model, tok, cap)
            record["compare"][cond] = A.compare(clean_cap, cap)
            del cap
        torch.save(record, path)
        del clean_cap
        torch.cuda.empty_cache()


# ── Gold-answer readout ──────────────────────────────────────────────────

def format_like(value: int, model_answer: str) -> str:
    """Print an integer dollar amount in the style of the model's answer."""
    return f"{value:,}" if "," in model_answer else str(value)


def first_divergence(gold_ids: list[int], pred_ids: list[int]) -> int | None:
    """First answer-token position where gold and model answer differ.

    Up to that position both answers share the same context, so it is where
    the model "chooses" wrong. None if the answers are identical.
    """
    for j, (g, p) in enumerate(zip(gold_ids, pred_ids)):
        if g != p:
            return j
    if len(gold_ids) != len(pred_ids):
        return min(len(gold_ids), len(pred_ids))
    return None


def run_gold_capture(model, tok, generations: list[dict], distractors: dict[str, str],
                     out_dir: Path, prompt_mode: str = "one_shot", top_k: int = 5) -> None:
    """Add a gold-answer readout to every capture record that lacks one.

    Replays each condition's prefix followed by the *gold* answer and records,
    per layer, the probability and rank of every gold token. At the divergence
    position the model's own token is tracked too (`alt_*`), so the layer where
    the wrong digit overtakes the right one can be located.
    """
    by_inst: dict[str, dict] = {}
    for r in generations:
        by_inst.setdefault(r["instance_id"], {"reasoning": {}})
        if r["scenario"] == "clean":
            by_inst[r["instance_id"]]["clean"] = r
        elif r["scenario"] == "reasoning" and r["variant"] != "control":
            by_inst[r["instance_id"]]["reasoning"][r["variant"]] = r
    after_answer = tok(".", add_special_tokens=False).input_ids[0]

    for path in tqdm(sorted(out_dir.glob("*.pt")), desc="gold capture"):
        rec = torch.load(path, weights_only=False)
        if "gold" in rec:
            continue
        d = by_inst[rec["instance_id"]]
        clean = d["clean"]
        gold_str = format_like(clean["gold"], rec["answer"])
        gold_ids = tok(gold_str, add_special_tokens=False).input_ids
        pred_ids = tok(rec["answer"], add_special_tokens=False).input_ids
        j = first_divergence(gold_ids, pred_ids)
        # alternative token per gold position: the model's own token where it exists
        alt = [pred_ids[k] if k < len(pred_ids) else after_answer for k in range(len(gold_ids))]
        alt_ids = torch.tensor(alt, device=A.output_device(model))

        prefixes = teacher_forced_prefixes(tok, clean, distractors, d["reasoning"], prompt_mode)
        runs = {"clean": (prefixes["clean"], 0), "clean_padded": (prefixes["clean"], NOISE_LEFT_PAD)}
        runs.update({c: (p, 0) for c, p in prefixes.items() if c != "clean"})
        conds = {}
        for cond, (prefix, pad) in runs.items():
            cap = A.capture(model, tok, prefix, gold_str, left_pad=pad)
            lens = A.logit_lens(model, cap, top_k=top_k, alt_ids=alt_ids)
            conds[cond] = {k: lens[k] for k in ("target_logp", "target_rank", "alt_logp",
                                                  "alt_rank", "top_ids", "top_logp")}
            del cap
        rec["gold"] = {"answer": gold_str, "value": clean["gold"],
                       "tokens": [tok.decode([t]) for t in gold_ids],
                       "pred_tokens": [tok.decode([t]) for t in pred_ids],
                       "alt_tokens": [tok.decode([t]) for t in alt],
                       "divergence": j, "conditions": conds}
        torch.save(rec, path)
        torch.cuda.empty_cache()


def load_captures(out_dir: Path) -> list[dict]:
    return [torch.load(p, weights_only=False) for p in sorted(out_dir.glob("*.pt"))]
