"""Test suite for the corrected evaluator (v2.1_corrected).

Covers the required categories: parsing, missing values, metrics,
statistics/IO. All tests must pass before corrected results are interpreted.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator_v2_1 import (  # noqa: E402
    EVALUATOR_VERSION,
    money_str_to_cents,
    parse_airline_final_answer,
    evaluate_response,
    is_missing,
    to_valid_cents,
    original_verdict,
    classify_causal_pair,
    classify_noncausal_pair,
    wilson_interval,
    metric_block,
    check_unique_ids,
    build_instance_record,
    aggregate_run,
)


def P(text, *, inside=False, completed=True):
    return parse_airline_final_answer(
        text, starts_inside_reasoning=inside, generation_completed=completed)


def R(text, finish="stop", inside=False):
    return evaluate_response(text, finish, starts_inside_reasoning=inside)


# ═══ PARSING ═══════════════════════════════════════════════════════════

def test_integer_dollar_total():
    r = P("Adding up: fees.\nThe total cost is $1245.")
    assert r.parse_status == "OK" and r.value_cents == 124500


def test_thousands_separator():
    assert P("The total cost is $1,245.").value_cents == 124500


def test_cents_00():
    assert P("The total cost is $1,245.00.").value_cents == 124500


def test_cents_nonzero():
    assert P("The total cost is $1,245.50.").value_cents == 124550


def test_single_decimal_digit():
    assert P("The total cost is $1245.5.").value_cents == 124550


def test_whitespace_after_dollar():
    assert P("The total cost is $ 1245.").value_cents == 124500


def test_bold_formatting():
    assert P("**The total cost is $1,245.**").value_cents == 124500


def test_case_insensitive_phrase():
    assert P("THE TOTAL COST IS $900.").value_cents == 90000


def test_repeated_identical_totals_not_conflicting():
    r = P("The total cost is $900.\nSo, the total cost is $900.")
    assert r.parse_status == "OK" and r.value_cents == 90000
    assert not r.conflicting_candidates


def test_conflicting_totals_completed_last_supersedes():
    r = P("The total cost is $800. Wait, I missed a bag.\n"
          "The total cost is $900.", completed=True)
    assert r.parse_status == "OK" and r.value_cents == 90000
    assert r.conflicting_candidates
    assert r.selection_rule == "last_statement_supersedes"


def test_conflicting_totals_truncated_is_ambiguous():
    r = P("The total cost is $800. Hmm.\nThe total cost is $900",
          completed=False)
    assert r.parse_status == "AMBIGUOUS_FINAL_ANSWER"
    assert r.value_cents is None


def test_intermediate_then_final_total():
    r = P("First pass: The total cost is $500. Recheck with fees...\n"
          "Final: the total cost is $650.")
    assert r.value_cents == 65000


def test_unfinished_reasoning_with_total_not_a_final_answer():
    r = P("Okay let me compute. The total cost is $500 so far, but wait...",
          inside=True)
    assert r.parse_status == "REASONING_UNCLOSED"
    assert r.value_cents is None


def test_reasoning_opened_in_prompt_closed_in_completion():
    r = P("thinking... tentatively The total cost is $500.\n</think>\n\n"
          "The total cost is $650.", inside=True)
    assert r.parse_status == "OK" and r.value_cents == 65000
    assert r.reasoning_closed is True


def test_reasoning_disabled_prefilled_empty_block():
    # Qwen3 enable_thinking=false: completion has no tags, all final region
    r = P("The total cost is $180 for the ticket... The total cost is $180.")
    assert r.parse_status == "OK" and r.value_cents == 18000


def test_explicit_think_block_in_completion_excluded():
    r = P("<think>The total cost is $111.</think>The total cost is $222.")
    assert r.value_cents == 22200


def test_truncated_without_final_answer():
    rec = R("I am computing the fees for bag 2 and", finish="length",
            inside=True)
    assert rec["strict_evaluable"] is False
    assert rec["exclusion_reason"] == "GENERATION_TRUNCATED"
    assert rec["aux_answer_but_truncated"] is False


def test_truncated_with_final_answer_is_auxiliary_only():
    rec = R("done.\n</think>\nThe total cost is $650.\nLet me also note",
            finish="length", inside=True)
    assert rec["strict_evaluable"] is False
    assert rec["aux_answer_but_truncated"] is True
    assert rec["value_cents"] == 65000  # recorded, but excluded from strict


def test_malformed_grouping():
    r = P("The total cost is $1,24,500.")
    assert r.parse_status == "MALFORMED_VALUE"


def test_malformed_three_decimals():
    r = P("The total cost is $12.345.")
    assert r.parse_status == "MALFORMED_VALUE"


def test_no_answer_phrase():
    assert P("Total fees: $400.").parse_status == "NO_ANSWER_PHRASE"


def test_phrase_without_value():
    assert P("The total cost is unknown.").parse_status == "NO_VALUE_AFTER_PHRASE"


def test_arithmetic_expression_after_phrase():
    r = P("The total cost is $800 + $100 = $900.")
    assert r.value_cents == 90000
    assert r.selection_rule == "after_equals"


def test_arithmetic_multiplication():
    r = P("The total cost is $180 + $1,065 = $1,245.")
    assert r.value_cents == 124500


def test_multi_value_no_equals_ambiguous():
    r = P("The total cost is $800 or $900.")
    assert r.parse_status == "AMBIGUOUS_FINAL_ANSWER"


def test_prompt_demo_total_inside_reasoning_not_selected():
    # demo-style restatement quoted inside an unclosed reasoning region
    r = P("In the example, The total cost is $530. Now for our case...",
          inside=True)
    assert r.value_cents is None


def test_empty_response():
    assert P("").parse_status == "EMPTY_RESPONSE"


def test_parser_reproducible():
    t = "The total cost is $800 + $100 = $900."
    assert P(t).to_dict() == P(t).to_dict()


# ═══ MONEY ═════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw,cents", [
    ("1,245", 124500), ("1,245.00", 124500), ("1,245.50", 124550),
    ("1245", 124500), ("1245.5", 124550), ("0", 0), ("999,999.99", 99999999),
])
def test_money_to_cents(raw, cents):
    assert money_str_to_cents(raw) == cents


@pytest.mark.parametrize("raw", ["1,24", "12,3456", "1.234", "1,245.123", ""])
def test_money_malformed(raw):
    assert money_str_to_cents(raw) is None


def test_one_cent_difference_is_a_difference():
    a = money_str_to_cents("1,245.00")
    b = money_str_to_cents("1,245.01")
    assert a != b


# ═══ MISSING VALUES ════════════════════════════════════════════════════

@pytest.mark.parametrize("x", [None, float("nan"), pd.NA])
def test_is_missing_variants(x):
    assert is_missing(x)


def test_numpy_nan_missing():
    import numpy as np
    assert is_missing(np.nan)
    assert to_valid_cents(np.nan) is None


def test_nonfinite_rejected():
    assert to_valid_cents(float("inf")) is None


def test_valid_cf_with_missing_original_not_evaluable():
    out = classify_causal_pair(
        orig_valid=False, causal_valid=True, orig_correct=None,
        pred_orig_cents=None, pred_causal_cents=110000,
        gold_orig_cents=100000, gold_causal_cents=110000)
    assert out == "NOT_EVALUABLE"
    nc = classify_noncausal_pair(
        orig_valid=False, noncausal_valid=True,
        pred_orig_cents=None, pred_noncausal_cents=110000,
        gold_orig_cents=100000, gold_noncausal_cents=100000)
    assert nc == "NOT_EVALUABLE"


def test_missing_causal_answer_not_evaluable():
    out = classify_causal_pair(
        orig_valid=True, causal_valid=False, orig_correct=True,
        pred_orig_cents=100000, pred_causal_cents=None,
        gold_orig_cents=100000, gold_causal_cents=110000)
    assert out == "NOT_EVALUABLE"


def test_missing_generation_metadata_unknown():
    rec = R("The total cost is $900.", finish=None)
    assert rec["generation_completed"] is None
    assert rec["strict_evaluable"] is False
    assert rec["exclusion_reason"] == "COMPLETION_METADATA_UNKNOWN"


def test_missing_raw_artifact():
    rec = evaluate_response(None, None, starts_inside_reasoning=False)
    assert rec["parse_status"] == "UNVERIFIABLE_ARTIFACT"
    assert rec["exclusion_reason"] == "MISSING_RAW_ARTIFACT"


# ═══ METRICS ═══════════════════════════════════════════════════════════

def _pair(orig_ok, pc, po=100000, go=100000, gc=110000):
    return classify_causal_pair(
        orig_valid=True, causal_valid=True, orig_correct=orig_ok,
        pred_orig_cents=po, pred_causal_cents=pc,
        gold_orig_cents=go, gold_causal_cents=gc)


def test_orig_correct_causal_correct():
    assert _pair(True, 110000) == "CORRECT_TRANSITION"


def test_orig_correct_causal_unchanged():
    assert _pair(True, 100000) == "INSENSITIVE"


def test_orig_correct_causal_changed_wrong():
    assert _pair(True, 120000) == "RESPONSIVE_BUT_WRONG"


def test_orig_incorrect_causal_correct_is_not_applicable():
    out = classify_causal_pair(
        orig_valid=True, causal_valid=True, orig_correct=False,
        pred_orig_cents=99900, pred_causal_cents=110000,
        gold_orig_cents=100000, gold_causal_cents=110000)
    assert out == "NOT_APPLICABLE"  # never RESPONSIVE_BUT_WRONG


def test_noncausal_invariant_and_changed():
    inv = classify_noncausal_pair(
        orig_valid=True, noncausal_valid=True,
        pred_orig_cents=99900, pred_noncausal_cents=99900,
        gold_orig_cents=100000, gold_noncausal_cents=100000)
    chg = classify_noncausal_pair(
        orig_valid=True, noncausal_valid=True,
        pred_orig_cents=99900, pred_noncausal_cents=100000,
        gold_orig_cents=100000, gold_noncausal_cents=100000)
    assert inv == "INVARIANT" and chg == "SPURIOUS_SENSITIVITY"


def _resp(cents, valid=True):
    return {
        "strict_evaluable": valid, "value_cents": cents,
        "exclusion_reason": None if valid else "GENERATION_TRUNCATED",
        "aux_answer_but_truncated": False, "parse_status": "OK" if valid else "X",
    }


def _record(iid, po, pc, pn, go=100000, gc=110000, ovalid=True, cvalid=True,
            nvalid=True, clean=True):
    return build_instance_record(
        instance_id=iid,
        orig_resp=_resp(po, ovalid), causal_resp=_resp(pc, cvalid),
        noncausal_resp=_resp(pn, nvalid),
        gold_orig_cents=go, gold_causal_cents=gc, gold_noncausal_cents=go,
        causal_policy_invariant=clean, noncausal_offset_preserved=True)


def test_no_combined_success_without_both_counterfactuals():
    r = _record(0, 100000, 110000, None, nvalid=False)
    assert r["functionally_consistent"] is None  # never positive
    r2 = _record(1, 100000, None, 100000, cvalid=False)
    assert r2["functionally_consistent"] is None


def test_combined_success_requires_all_three():
    ok = _record(0, 100000, 110000, 100000)
    assert ok["functionally_consistent"] == 1
    bad_inv = _record(1, 100000, 110000, 99000)
    assert bad_inv["functionally_consistent"] == 0


def test_cohorts_expose_ids_not_just_sizes():
    recs = [
        _record(0, 100000, 110000, None, nvalid=False),  # S_c only
        _record(1, 100000, None, 100000, cvalid=False),  # S_n only
    ]
    agg = aggregate_run(recs)
    assert agg["cohorts"]["S_c"]["n"] == agg["cohorts"]["S_n"]["n"] == 1
    assert agg["cohorts"]["S_c"]["ids"] != agg["cohorts"]["S_n"]["ids"]
    assert agg["cohorts"]["J"]["n"] == 0
    assert agg["metrics"]["joint_consistency_J"]["not_estimable"]


def test_delta_identity_when_original_correct():
    # pred_orig == gold_orig -> delta match <=> causal correctness
    r_match = _record(0, 100000, 110000, 100000)
    assert r_match["delta_exact_match"] and r_match["causal_correct"]
    r_miss = _record(1, 100000, 120000, 100000)
    assert not r_miss["delta_exact_match"] and not r_miss["causal_correct"]
    agg = aggregate_run([r_match, r_miss])
    assert agg["delta_identity_violations"] == []


def test_decimal_formatting_must_not_create_spurious_sensitivity():
    a = P("The total cost is $1,245.00.").value_cents
    b = P("The total cost is $1245.").value_cents
    assert a == b  # same value differently formatted -> invariant
    nc = classify_noncausal_pair(
        orig_valid=True, noncausal_valid=True,
        pred_orig_cents=a, pred_noncausal_cents=b,
        gold_orig_cents=999900, gold_noncausal_cents=999900)
    assert nc == "INVARIANT"


def test_unconditioned_2x2_reports_incorrect_to_correct():
    recs = [_record(0, 99900, 110000, 99900)]  # orig wrong, causal right
    agg = aggregate_run(recs)
    combos = agg["unconditioned_causal_2x2_on_E_c"]["combos"]
    assert combos["orig_incorrect__causal_correct"]["n"] == 1


def test_delta_three_estimands_differ():
    recs = [
        _record(0, 99000, 109000, 99000),          # valid pair, delta match
        _record(1, 100000, None, 100000, cvalid=False),  # attempted, invalid
    ]
    agg = aggregate_run(recs)
    d = agg["delta"]["all_50"]
    assert d["DELTA_PAIR_COVERAGE"]["rate"] == 0.5
    assert d["DELTA_EXACT_MATCH_GIVEN_VALID_PAIR"]["rate"] == 1.0
    assert d["DELTA_SUCCESS_ALL_ATTEMPTS"]["rate"] == 0.5


# ═══ STATISTICS / IO ═══════════════════════════════════════════════════

def test_wilson_zero_over_n_nonzero_width():
    lo, hi = wilson_interval(0, 4)
    assert lo == 0.0 and hi > 0.3  # not [0,0]


def test_wilson_n_over_n_nonzero_width():
    lo, hi = wilson_interval(2, 2)
    assert hi == 1.0 and lo < 0.7  # not [1,1]


def test_wilson_n_zero_not_estimable():
    assert wilson_interval(0, 0) is None
    b = metric_block("x", 0, 0)
    assert b["rate"] is None and b["wilson95"] is None and b["not_estimable"]


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        check_unique_ids([1, 2, 2])


def test_currency_roundtrip_json():
    rec = R("The total cost is $1,245.50.")
    s = json.dumps(rec, allow_nan=False)
    back = json.loads(s)
    assert back["value_cents"] == 124550


def test_json_no_nan():
    rec = R("gibberish with no answer")
    json.dumps(rec, allow_nan=False)  # must not raise


def test_units_consistent_cents_everywhere():
    r = _record(0, 100000, 110000, 100000)
    assert r["delta_model_cents"] == 10000  # $100 in cents
    assert r["delta_gold_cents"] == 10000


def test_evaluator_version_recorded():
    rec = R("The total cost is $900.")
    assert rec["evaluator_version"] == EVALUATOR_VERSION == "v2.1_corrected"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ═══ ADAPTIVE (design absence) ═════════════════════════════════════════

@pytest.mark.skip(reason="depende de scripts/10_reevaluate_v2_1.py, do estudo de grounding funcional, fora deste repositório")
def test_designed_absence_is_not_evaluable_and_never_positive():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "reeval", Path(__file__).resolve().parents[1] /
        "scripts" / "10_reevaluate_v2_1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    absent = mod.designed_absence_record()
    assert absent["strict_evaluable"] is False
    assert absent["exclusion_reason"] == "NOT_GENERATED_BY_DESIGN"
    rec = build_instance_record(
        instance_id=0,
        orig_resp=_resp(100000), causal_resp=absent, noncausal_resp=absent,
        gold_orig_cents=100000, gold_causal_cents=110000,
        gold_noncausal_cents=100000,
        causal_policy_invariant=True, noncausal_offset_preserved=True)
    assert rec["transition_class"] == "NOT_EVALUABLE"
    assert rec["noncausal_class"] == "NOT_EVALUABLE"
    assert rec["E_c"] is False and rec["E_n"] is False
    assert rec["functionally_consistent"] is None
    assert rec["causal_exclusion"] == "NOT_GENERATED_BY_DESIGN"
