"""Corrected evaluator for RuleArena V2 Airline runs.

Single authoritative implementation of the frozen specification in
outputs/evaluator_v2_1/EVALUATOR_CORRECTION_SPEC.md.

    evaluator_version = "v2.1_corrected"

Design rules (see spec):
  - final-answer region determined by reasoning-state segmentation;
  - monetary values as integer cents via Decimal (never float, never
    separator stripping);
  - completion / extraction / correctness recorded independently;
  - explicit nullable fields; NaN never reaches a comparison;
  - the parser never sees gold, intervention labels, or expected deltas.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from decimal import Decimal, InvalidOperation

EVALUATOR_VERSION = "v2.1_corrected"

ANSWER_PHRASE_RE = re.compile(r"the\s+total\s+cost\s+is", re.IGNORECASE)
MONEY_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
THINK_TAG_RE = re.compile(r"</?think>")
STATEMENT_WINDOW = 120

PARSE_OK = "OK"
PARSE_NO_PHRASE = "NO_ANSWER_PHRASE"
PARSE_NO_VALUE = "NO_VALUE_AFTER_PHRASE"
PARSE_MALFORMED = "MALFORMED_VALUE"
PARSE_AMBIGUOUS = "AMBIGUOUS_FINAL_ANSWER"
PARSE_REASONING_UNCLOSED = "REASONING_UNCLOSED"
PARSE_EMPTY = "EMPTY_RESPONSE"


# ── money ────────────────────────────────────────────────────────────────

def money_str_to_cents(raw: str) -> int | None:
    """Parse a monetary token (digits, optional commas, optional decimals)
    into integer cents. Returns None for malformed values (bad grouping,
    >2 decimal digits, non-finite)."""
    s = raw.strip()
    if not s:
        return None
    parts = s.split(".")
    if len(parts) > 2:
        return None
    int_part = parts[0]
    dec_part = parts[1] if len(parts) == 2 else ""
    if "," in int_part:
        if not re.fullmatch(r"\d{1,3}(,\d{3})+", int_part):
            return None  # malformed grouping
        int_part = int_part.replace(",", "")
    if not int_part.isdigit():
        return None
    if dec_part:
        if not dec_part.isdigit() or len(dec_part) > 2:
            return None
    try:
        value = Decimal(int_part + ("." + dec_part if dec_part else ""))
    except InvalidOperation:
        return None
    cents = value * 100
    if cents != cents.to_integral_value():
        return None
    return int(cents)


def cents_to_dollars_str(cents: int | None) -> str | None:
    if cents is None:
        return None
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


# ── final-answer region ─────────────────────────────────────────────────

def split_final_region(text: str, starts_inside_reasoning: bool
                       ) -> tuple[list[tuple[int, str]], bool, bool]:
    """Segment the completion into final-answer-region segments.

    Returns (segments, reasoning_closed, had_reasoning) where segments is a
    list of (start_offset, segment_text) OUTSIDE reasoning blocks.
    reasoning_closed is False iff the scan ends inside a reasoning block.
    """
    segments: list[tuple[int, str]] = []
    inside = starts_inside_reasoning
    had_reasoning = starts_inside_reasoning
    pos = 0
    for m in THINK_TAG_RE.finditer(text):
        if m.group() == "<think>":
            if not inside:
                if pos < m.start():
                    segments.append((pos, text[pos:m.start()]))
                inside = True
                had_reasoning = True
            # nested/dangling <think> while inside: stay inside
        else:  # </think>
            if inside:
                inside = False
            # stray </think> while outside: ignore, remain outside
        pos = m.end()
    if not inside and pos <= len(text):
        seg = text[pos:]
        if seg:
            segments.append((pos, seg))
    return segments, (not inside), had_reasoning


# ── statement extraction ────────────────────────────────────────────────

@dataclass
class AnswerStatement:
    abs_start: int                     # offset of the phrase in the completion
    window_text: str
    candidates_cents: list[int | None]  # None marks a malformed token
    candidate_spans: list[tuple[int, int]]
    selected_cents: int | None
    selection_rule: str                # single_value | after_equals |
    #                                    ambiguous_multi_value | malformed | none


def _extract_statement(segment: str, seg_offset: int,
                       phrase_end: int) -> AnswerStatement:
    window_end = phrase_end + STATEMENT_WINDOW
    nl = segment.find("\n", phrase_end)
    if nl != -1:
        window_end = min(window_end, nl)
    window = segment[phrase_end:window_end]

    tokens = list(MONEY_RE.finditer(window))
    cents = [money_str_to_cents(t.group(1)) for t in tokens]
    spans = [(seg_offset + phrase_end + t.start(),
              seg_offset + phrase_end + t.end()) for t in tokens]

    if not tokens:
        sel, rule = None, "none"
    elif len(tokens) == 1:
        sel = cents[0]
        rule = "single_value" if sel is not None else "malformed"
    else:
        eq = window.rfind("=")
        if eq != -1:
            after = [c for t, c in zip(tokens, cents) if t.start() > eq]
            if after:
                sel = after[-1]
                rule = "after_equals" if sel is not None else "malformed"
            else:
                sel, rule = None, "ambiguous_multi_value"
        else:
            sel, rule = None, "ambiguous_multi_value"
    return AnswerStatement(seg_offset + phrase_end - 0, window,
                           cents, spans, sel, rule)


# ── parser ───────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    parse_status: str
    value_cents: int | None = None
    selected_span: tuple[int, int] | None = None
    selection_rule: str | None = None
    conflicting_candidates: bool = False
    n_answer_statements: int = 0
    candidate_values_cents: list = field(default_factory=list)
    final_region_segments: list = field(default_factory=list)  # (start,end)
    reasoning_closed: bool | None = None
    had_reasoning: bool = False
    evaluator_version: str = EVALUATOR_VERSION

    def to_dict(self):
        d = asdict(self)
        d["value_dollars"] = cents_to_dollars_str(self.value_cents)
        return d


def parse_airline_final_answer(text: str, *, starts_inside_reasoning: bool,
                               generation_completed: bool | None
                               ) -> ParseResult:
    """Extract the final total-cost answer per the frozen v2.1 spec."""
    if not text or not text.strip():
        return ParseResult(PARSE_EMPTY, reasoning_closed=None)

    segments, closed, had_reasoning = split_final_region(
        text, starts_inside_reasoning)
    seg_bounds = [(off, off + len(seg)) for off, seg in segments]

    if not any(seg.strip() for _, seg in segments):
        status = (PARSE_REASONING_UNCLOSED if not closed
                  else PARSE_NO_PHRASE)
        return ParseResult(status, final_region_segments=seg_bounds,
                           reasoning_closed=closed, had_reasoning=had_reasoning)

    statements: list[AnswerStatement] = []
    for off, seg in segments:
        clean = seg.replace("**", "")
        # build an offset map is overkill: '**' removal shifts spans; keep
        # spans approximate by searching on the cleaned text but recording
        # cleaned-relative offsets. For audit we store the window text itself.
        for m in ANSWER_PHRASE_RE.finditer(clean):
            statements.append(_extract_statement(clean, off, m.end()))

    base = ParseResult(PARSE_OK, final_region_segments=seg_bounds,
                       reasoning_closed=closed, had_reasoning=had_reasoning,
                       n_answer_statements=len(statements))
    base.candidate_values_cents = [s.candidates_cents for s in statements]

    if not statements:
        base.parse_status = PARSE_NO_PHRASE
        return base

    values = [s.selected_cents for s in statements]
    rules = [s.selection_rule for s in statements]

    if all(v is None for v in values):
        # no statement produced a usable value
        if any(r == "ambiguous_multi_value" for r in rules):
            base.parse_status = PARSE_AMBIGUOUS
        elif any(r == "malformed" for r in rules):
            base.parse_status = PARSE_MALFORMED
        else:
            base.parse_status = PARSE_NO_VALUE
        return base

    distinct = {v for v in values if v is not None}
    last = statements[-1]

    if len(distinct) == 1 and last.selected_cents is not None:
        base.value_cents = distinct.pop()
        base.selected_span = last.candidate_spans[-1] if last.candidate_spans else None
        base.selection_rule = last.selection_rule
        return base

    # conflicting or last-statement problems
    base.conflicting_candidates = len(distinct) > 1
    if last.selected_cents is None:
        base.parse_status = PARSE_AMBIGUOUS
        return base
    if len(distinct) > 1:
        if generation_completed:
            base.value_cents = last.selected_cents
            base.selected_span = (last.candidate_spans[-1]
                                  if last.candidate_spans else None)
            base.selection_rule = "last_statement_supersedes"
            return base
        base.parse_status = PARSE_AMBIGUOUS
        return base
    # single distinct value but produced by a non-last statement while the
    # last statement had no tokens at all: accept the distinct value
    base.value_cents = distinct.pop()
    base.selection_rule = "single_value"
    return base


# ── response record ─────────────────────────────────────────────────────

def evaluate_response(text: str | None, finish_reason: str | None, *,
                      starts_inside_reasoning: bool) -> dict:
    """Full per-response record: completion, extraction, evaluability."""
    if text is None:
        return {
            "native_finish_reason": finish_reason,
            "generation_completed": None,
            "reached_generation_limit": None,
            "reasoning_closed": None,
            "explicit_final_answer_present": False,
            "parse_status": "UNVERIFIABLE_ARTIFACT",
            "value_cents": None,
            "strict_evaluable": False,
            "exclusion_reason": "MISSING_RAW_ARTIFACT",
            "aux_answer_but_truncated": False,
            "parse_detail": None,
            "evaluator_version": EVALUATOR_VERSION,
        }

    if finish_reason == "stop":
        completed: bool | None = True
    elif finish_reason == "length":
        completed = False
    else:
        completed = None  # UNKNOWN — never assumed complete

    pr = parse_airline_final_answer(
        text, starts_inside_reasoning=starts_inside_reasoning,
        generation_completed=bool(completed))

    answer_present = pr.parse_status == PARSE_OK
    strict = (completed is True) and (pr.reasoning_closed is not False) \
        and answer_present

    if strict:
        exclusion = None
    elif completed is False:
        exclusion = "GENERATION_TRUNCATED"
    elif completed is None:
        exclusion = "COMPLETION_METADATA_UNKNOWN"
    elif pr.reasoning_closed is False:
        exclusion = "REASONING_UNCLOSED"
    else:
        exclusion = f"PARSE_{pr.parse_status}"

    return {
        "native_finish_reason": finish_reason,
        "generation_completed": completed,
        "reached_generation_limit": finish_reason == "length",
        "reasoning_closed": pr.reasoning_closed,
        "explicit_final_answer_present": answer_present,
        "parse_status": pr.parse_status,
        "value_cents": pr.value_cents if strict or answer_present else pr.value_cents,
        "strict_evaluable": strict,
        "exclusion_reason": exclusion,
        "aux_answer_but_truncated": (completed is False) and answer_present,
        "parse_detail": pr.to_dict(),
        "evaluator_version": EVALUATOR_VERSION,
    }


# ── correctness / classification ────────────────────────────────────────

def is_missing(x) -> bool:
    """Uniform missing test for None / float NaN / numpy NaN / pandas NA."""
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    try:
        import pandas as pd
        if x is pd.NA or (hasattr(pd, "isna") and not isinstance(x, (list, dict))
                          and pd.isna(x)):
            return True
    except Exception:
        pass
    return False


def to_valid_cents(x) -> int | None:
    """Normalize any ingested numeric to int cents or None (non-finite→None)."""
    if is_missing(x):
        return None
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        if x != int(x):
            return None
        return int(x)
    return int(x)


def original_verdict(resp: dict, gold_cents: int) -> str:
    if resp["parse_status"] == "UNVERIFIABLE_ARTIFACT":
        return "UNVERIFIABLE_ARTIFACT"
    if not resp["strict_evaluable"]:
        return "NO_EVALUABLE_FINAL_ANSWER"
    return ("CORRECT_FINAL_ANSWER"
            if resp["value_cents"] == gold_cents
            else "INCORRECT_FINAL_ANSWER")


def classify_causal_pair(*, orig_valid: bool, causal_valid: bool,
                         orig_correct: bool | None,
                         pred_orig_cents: int | None,
                         pred_causal_cents: int | None,
                         gold_orig_cents: int, gold_causal_cents: int) -> str:
    """Primary conditioned taxonomy (spec §6). Returns one of
    CORRECT_TRANSITION / INSENSITIVE / RESPONSIVE_BUT_WRONG /
    NOT_APPLICABLE / NOT_EVALUABLE."""
    if not (orig_valid and causal_valid):
        return "NOT_EVALUABLE"
    if gold_causal_cents == gold_orig_cents:
        return "NOT_EVALUABLE"  # invalid intervention (data problem)
    if not orig_correct:
        return "NOT_APPLICABLE"
    assert pred_orig_cents is not None and pred_causal_cents is not None
    if pred_causal_cents == gold_causal_cents:
        return "CORRECT_TRANSITION"
    if pred_causal_cents == pred_orig_cents:
        return "INSENSITIVE"
    return "RESPONSIVE_BUT_WRONG"


def classify_noncausal_pair(*, orig_valid: bool, noncausal_valid: bool,
                            pred_orig_cents: int | None,
                            pred_noncausal_cents: int | None,
                            gold_orig_cents: int,
                            gold_noncausal_cents: int) -> str:
    """Non-causal behavior on E_n (spec §6). Returns
    INVARIANT / SPURIOUS_SENSITIVITY / NOT_EVALUABLE."""
    if not (orig_valid and noncausal_valid):
        return "NOT_EVALUABLE"
    if gold_noncausal_cents != gold_orig_cents:
        return "NOT_EVALUABLE"  # gold not preserved — data problem
    assert pred_orig_cents is not None and pred_noncausal_cents is not None
    return ("INVARIANT" if pred_noncausal_cents == pred_orig_cents
            else "SPURIOUS_SENSITIVITY")


# ── statistics ───────────────────────────────────────────────────────────

def wilson_interval(k: int, n: int, z: float = 1.959963984540054
                    ) -> tuple[float, float] | None:
    """Wilson 95% score interval for a binomial proportion. n=0 → None."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def metric_block(name: str, k: int, n: int, ids: list[int] | None = None,
                 exclusions: dict | None = None) -> dict:
    ci = wilson_interval(k, n)
    return {
        "metric": name,
        "numerator": k,
        "denominator": n,
        "rate": (k / n) if n > 0 else None,
        "wilson95": list(ci) if ci else None,
        "not_estimable": n == 0,
        "eligible_ids": sorted(ids) if ids is not None else None,
        "exclusions": exclusions,
    }


# ── run-level aggregation ────────────────────────────────────────────────

def check_unique_ids(ids: list) -> None:
    """Raise on duplicate IDs — alignment must never fall back to row order."""
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate instance IDs: {dupes}")


def build_instance_record(*, instance_id: int,
                          orig_resp: dict, causal_resp: dict,
                          noncausal_resp: dict,
                          gold_orig_cents: int, gold_causal_cents: int,
                          gold_noncausal_cents: int,
                          causal_policy_invariant: bool,
                          noncausal_offset_preserved: bool | None) -> dict:
    """Assemble the per-instance corrected record (spec §6-§9)."""
    o_valid = orig_resp["strict_evaluable"]
    c_valid = causal_resp["strict_evaluable"]
    n_valid = noncausal_resp["strict_evaluable"]
    verdict = original_verdict(orig_resp, gold_orig_cents)
    o_correct = verdict == "CORRECT_FINAL_ANSWER"

    po = orig_resp["value_cents"] if o_valid else None
    pc = causal_resp["value_cents"] if c_valid else None
    pn = noncausal_resp["value_cents"] if n_valid else None

    transition = classify_causal_pair(
        orig_valid=o_valid, causal_valid=c_valid, orig_correct=o_correct,
        pred_orig_cents=po, pred_causal_cents=pc,
        gold_orig_cents=gold_orig_cents, gold_causal_cents=gold_causal_cents)
    nc_class = classify_noncausal_pair(
        orig_valid=o_valid, noncausal_valid=n_valid,
        pred_orig_cents=po, pred_noncausal_cents=pn,
        gold_orig_cents=gold_orig_cents,
        gold_noncausal_cents=gold_noncausal_cents)

    E_c = o_valid and c_valid
    E_n = o_valid and n_valid
    rec = {
        "instance_id": instance_id,
        "original_verdict": verdict,
        "original_valid": o_valid,
        "original_correct": o_correct if o_valid else None,
        "causal_valid": c_valid,
        "noncausal_valid": n_valid,
        "pred_original_cents": po,
        "pred_causal_cents": pc,
        "pred_noncausal_cents": pn,
        "gold_original_cents": gold_orig_cents,
        "gold_causal_cents": gold_causal_cents,
        "gold_noncausal_cents": gold_noncausal_cents,
        "E_c": E_c,
        "E_n": E_n,
        "causal_correct": (pc == gold_causal_cents) if E_c else None,
        "prediction_changed_causal": (pc != po) if E_c else None,
        "prediction_changed_noncausal": (pn != po) if E_n else None,
        "transition_class": transition,
        "noncausal_class": nc_class,
        "delta_model_cents": (pc - po) if E_c else None,
        "delta_gold_cents": gold_causal_cents - gold_orig_cents,
        "delta_exact_match": ((pc - po) == (gold_causal_cents - gold_orig_cents)
                              ) if E_c else None,
        "causal_policy_invariant": causal_policy_invariant,
        "noncausal_offset_preserved": noncausal_offset_preserved,
        "orig_exclusion": orig_resp["exclusion_reason"],
        "causal_exclusion": causal_resp["exclusion_reason"],
        "noncausal_exclusion": noncausal_resp["exclusion_reason"],
        "aux_orig_answer_but_truncated": orig_resp["aux_answer_but_truncated"],
        "aux_causal_answer_but_truncated": causal_resp["aux_answer_but_truncated"],
        "aux_noncausal_answer_but_truncated": noncausal_resp["aux_answer_but_truncated"],
    }
    # combined consistency: only computable when both arms evaluable (J);
    # missing either counterfactual must never yield a positive
    if o_valid and o_correct and E_c and E_n:
        rec["functionally_consistent"] = int(
            transition == "CORRECT_TRANSITION" and nc_class == "INVARIANT")
    else:
        rec["functionally_consistent"] = None
    return rec


def aggregate_run(records: list[dict]) -> dict:
    """Run-level cohorts and metric blocks per spec §7-§9."""
    check_unique_ids([r["instance_id"] for r in records])
    n_att = len(records)
    by_id = {r["instance_id"]: r for r in records}

    def ids(pred):
        return sorted(i for i, r in by_id.items() if pred(r))

    C0 = ids(lambda r: r["original_valid"] and r["original_correct"])
    S_c = ids(lambda r: r["instance_id"] in set(C0) and r["causal_valid"])
    S_n = ids(lambda r: r["instance_id"] in set(C0) and r["noncausal_valid"])
    J = sorted(set(S_c) & set(S_n))

    def excl(cohort_ids, base_ids, arm_key):
        return {str(i): by_id[i][arm_key]
                for i in base_ids if i not in set(cohort_ids)}

    out = {
        "n_attempted": n_att,
        "cohorts": {
            "C0": {"ids": C0, "n": len(C0)},
            "S_c": {"ids": S_c, "n": len(S_c),
                    "exclusions_from_C0": excl(S_c, C0, "causal_exclusion")},
            "S_n": {"ids": S_n, "n": len(S_n),
                    "exclusions_from_C0": excl(S_n, C0, "noncausal_exclusion")},
            "J": {"ids": J, "n": len(J)},
        },
        "metrics": {},
    }
    M = out["metrics"]

    # original-answer verdicts over all attempts
    verd = {}
    for r in records:
        verd[r["original_verdict"]] = verd.get(r["original_verdict"], 0) + 1
    out["original_verdict_counts"] = verd
    M["original_accuracy_all_attempts"] = metric_block(
        "original_accuracy_all_attempts", len(C0), n_att, C0)
    n_orig_valid = sum(1 for r in records if r["original_valid"])
    M["original_accuracy_given_valid"] = metric_block(
        "original_accuracy_given_valid", len(C0), n_orig_valid,
        ids(lambda r: r["original_valid"]))
    M["original_strict_evaluable_rate"] = metric_block(
        "original_strict_evaluable_rate", n_orig_valid, n_att)

    # conditioned causal taxonomy on S_c
    def tax_count(cls):
        return sum(1 for i in S_c if by_id[i]["transition_class"] == cls)
    for cls, name in [("CORRECT_TRANSITION", "transition_accuracy"),
                      ("INSENSITIVE", "insensitivity"),
                      ("RESPONSIVE_BUT_WRONG", "responsive_but_wrong")]:
        M[f"{name}_S_c"] = metric_block(f"{name}_S_c", tax_count(cls),
                                        len(S_c), S_c)

    # non-causal on S_n
    inv_Sn = sum(1 for i in S_n if by_id[i]["noncausal_class"] == "INVARIANT")
    M["noncausal_invariance_S_n"] = metric_block(
        "noncausal_invariance_S_n", inv_Sn, len(S_n), S_n)
    M["spurious_sensitivity_S_n"] = metric_block(
        "spurious_sensitivity_S_n", len(S_n) - inv_Sn, len(S_n), S_n)

    # both on J (paired comparison basis)
    ct_J = sum(1 for i in J if by_id[i]["transition_class"] == "CORRECT_TRANSITION")
    inv_J = sum(1 for i in J if by_id[i]["noncausal_class"] == "INVARIANT")
    M["transition_accuracy_J"] = metric_block("transition_accuracy_J", ct_J,
                                              len(J), J)
    M["noncausal_invariance_J"] = metric_block("noncausal_invariance_J", inv_J,
                                               len(J), J)
    # paired discordance on J
    out["J_paired_table"] = {
        "ct_and_inv": sum(1 for i in J
                          if by_id[i]["transition_class"] == "CORRECT_TRANSITION"
                          and by_id[i]["noncausal_class"] == "INVARIANT"),
        "ct_only": sum(1 for i in J
                       if by_id[i]["transition_class"] == "CORRECT_TRANSITION"
                       and by_id[i]["noncausal_class"] != "INVARIANT"),
        "inv_only": sum(1 for i in J
                        if by_id[i]["transition_class"] != "CORRECT_TRANSITION"
                        and by_id[i]["noncausal_class"] == "INVARIANT"),
        "neither": sum(1 for i in J
                       if by_id[i]["transition_class"] != "CORRECT_TRANSITION"
                       and by_id[i]["noncausal_class"] != "INVARIANT"),
    }

    # combined consistency
    fc = [r for r in records if r["functionally_consistent"] is not None]
    n_fc_pos = sum(r["functionally_consistent"] for r in fc)
    M["joint_consistency_J"] = metric_block(
        "joint_consistency_J", n_fc_pos, len(J), J)
    M["joint_delivery_all_attempts"] = metric_block(
        "joint_delivery_all_attempts", n_fc_pos, n_att)

    # unconditioned E_c 2x2
    Ec_ids = ids(lambda r: r["E_c"])
    combos = {}
    for r in (by_id[i] for i in Ec_ids):
        key = (f"orig_{'correct' if r['original_correct'] else 'incorrect'}"
               f"__causal_{'correct' if r['causal_correct'] else 'incorrect'}")
        combos.setdefault(key, {"n": 0, "changed": 0})
        combos[key]["n"] += 1
        combos[key]["changed"] += int(r["prediction_changed_causal"])
    out["unconditioned_causal_2x2_on_E_c"] = {
        "n_E_c": len(Ec_ids), "ids_E_c": Ec_ids, "combos": combos}

    # unconditioned non-causal on E_n
    En_ids = ids(lambda r: r["E_n"])
    inv_all = sum(1 for i in En_ids
                  if by_id[i]["noncausal_class"] == "INVARIANT")
    M["noncausal_invariance_all_valid_E_n"] = metric_block(
        "noncausal_invariance_all_valid_E_n", inv_all, len(En_ids), En_ids)

    # delta metrics (spec §8), with policy subsets (spec §9)
    def delta_blocks(tag, subset_pred):
        att = [r for r in records if subset_pred(r)]
        valid = [r for r in att if r["E_c"]]
        match = [r for r in valid if r["delta_exact_match"]]
        blocks = {
            "DELTA_PAIR_COVERAGE": metric_block(
                f"{tag}_coverage", len(valid), len(att),
                [r["instance_id"] for r in valid]),
            "DELTA_EXACT_MATCH_GIVEN_VALID_PAIR": metric_block(
                f"{tag}_match_given_valid", len(match), len(valid),
                [r["instance_id"] for r in match]),
            "DELTA_SUCCESS_ALL_ATTEMPTS": metric_block(
                f"{tag}_success_all_attempts", len(match), len(att)),
        }
        errs = [abs(r["delta_model_cents"] - r["delta_gold_cents"]) / 100
                for r in valid]
        blocks["mean_abs_delta_error_dollars_valid_pairs"] = (
            sum(errs) / len(errs) if errs else None)
        blocks["median_abs_delta_error_dollars_valid_pairs"] = (
            sorted(errs)[len(errs) // 2] if errs else None)
        # strata among valid pairs
        for label, cond in [("valid_orig_correct", True),
                            ("valid_orig_incorrect", False)]:
            sub = [r for r in valid if r["original_correct"] is cond]
            m = sum(1 for r in sub if r["delta_exact_match"])
            blocks[label] = metric_block(f"{tag}_{label}", m, len(sub),
                                         [r["instance_id"] for r in sub])
        return blocks

    out["delta"] = {
        "all_50": delta_blocks("all", lambda r: True),
        "policy_invariant_causal": delta_blocks(
            "clean", lambda r: r["causal_policy_invariant"]),
        "policy_ambiguous_causal": delta_blocks(
            "ambig", lambda r: not r["causal_policy_invariant"]),
    }

    # identity check (spec §8): among valid pairs with correct original,
    # delta_exact_match must equal causal_correct
    viol = [r["instance_id"] for r in records
            if r["E_c"] and r["original_correct"]
            and bool(r["delta_exact_match"]) != bool(r["causal_correct"])]
    out["delta_identity_violations"] = viol

    return out
