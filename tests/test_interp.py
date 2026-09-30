"""Tests for irrelevant-information insertion and convergence detection (src/interp)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interp.activations import convergence_layer  # noqa: E402
from src.interp.distractors import (  # noqa: E402
    find_reasoning_cut, insert_in_question, insert_in_reasoning, split_final_answer,
)
from src.interp.experiment import first_divergence, format_like  # noqa: E402

QUESTION = (
    "Sarah is a Main Cabin Class passenger flying from Orlando to Philadelphia with the following items:\n"
    "1. A backpack: 22 x 13 x 6 inches, 10 lbs;\n"
    "2. A luggage box: 44 x 22 x 20 inches, 69 lbs;\n"
    "3. A luggage box: 34 x 18 x 12 inches, 51 lbs;\n"
    "4. A backpack: 38 x 22 x 16 inches, 84 lbs;\n"
    "\nSarah's flight ticket is $180."
)


def test_insert_in_question_after_middle_item():
    out = insert_in_question(QUESTION, "I stopped to breathe.")
    lines = out.splitlines()
    assert lines[2].startswith("2. ") and lines[3] == "I stopped to breathe."
    assert out.replace("\nI stopped to breathe.", "") == QUESTION   # nothing else changed


def test_reasoning_cut_lands_inside_a_sum():
    text = ("Item 1: fee $0.\n" * 20 + "Total Fee for Item 2: $35 (checking) + $150 (oversize) = $185\n"
            "The total cost is $365.")
    cut, kind = find_reasoning_cut(text)
    assert kind == "plus"
    assert text[:cut].endswith("$35 (checking) + ")
    assert text[cut:].startswith("$150")


def test_reasoning_cut_never_uses_the_answer_sentence():
    text = "No arithmetic here.\nStill none.\nThe total cost is $1 + $2."
    cut, kind = find_reasoning_cut(text, min_frac=0.0)
    assert kind == "newline" and cut < text.index("The total cost")


def test_insert_in_reasoning_spacing():
    assert insert_in_reasoning("$35 + ", "I stopped to breathe.") == "$35 + I stopped to breathe. "
    assert insert_in_reasoning("$35 +", "x.") == "$35 + x. "


def test_split_final_answer_uses_last_occurrence():
    text = "The total cost is $10. Wait. The total cost is $1,365."
    prefix, answer = split_final_answer(text)
    assert answer == "1,365" and prefix.endswith("The total cost is $")
    assert split_final_answer("no answer") is None


def test_split_final_answer_accepts_markdown_bold():
    text = "- Total Baggage Fees: $1,150\n- The total cost is **$1,434**."
    prefix, answer = split_final_answer(text)
    assert answer == "1,434" and prefix.endswith("The total cost is **$")
    cut, _ = find_reasoning_cut("Fee: $35 + $150 = $185\n" * 5 + "The total cost is **$1 + $2**.")
    assert cut < len("Fee: $35 + $150 = $185\n" * 5)      # the answer sentence is still never a cut point


@pytest.mark.parametrize("ranks,expected", [
    ([[3, 2], [0, 5], [0, 0], [0, 0]], 2),   # both digits top-1 from layer 2 on
    ([[0, 0], [1, 0], [0, 0], [0, 0]], 2),   # an early top-1 that is lost does not count
    ([[0, 0], [0, 0], [0, 0]], 0),
    ([[0, 0], [0, 0], [0, 1]], None),        # final layer would not produce this answer
])
def test_convergence_layer(ranks, expected):
    assert convergence_layer(np.array(ranks)) == expected



def test_format_like_follows_model_style():
    assert format_like(1245, "1,365") == "1,245"
    assert format_like(1245, "1365") == "1245"
    assert format_like(992, "1,365") == "992"


@pytest.mark.parametrize("gold,pred,expected", [
    ([1, 2, 4, 5], [1, 3, 6, 5], 1),   # first differing digit
    ([1, 2, 4], [1, 2, 4], None),      # identical answers
    ([1, 2, 4, 5], [1, 2, 4], 3),      # model stops early
    ([9, 9], [9, 9, 2], 2),            # model writes an extra digit
])
def test_first_divergence(gold, pred, expected):
    assert first_divergence(gold, pred) == expected


# ── Greedy decoding ignores the model's generation_config ────────────────

def test_greedy_overrides_generation_config_penalties():
    """Qwen2.5 ships repetition_penalty=1.05; greedy decoding must not inherit it."""
    from transformers import GenerationConfig

    from src.interp.experiment import GREEDY
    shipped = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                               repetition_penalty=1.05, no_repeat_ngram_size=3)
    shipped.update(**GREEDY)          # what generate() does with its kwargs
    assert shipped.do_sample is False
    assert shipped.repetition_penalty == 1.0
    assert shipped.no_repeat_ngram_size == 0


def test_generate_greedy_passes_overrides():
    from types import SimpleNamespace

    import torch

    from src.interp import experiment as E
    seen = {}

    class Enc(dict):
        __getattr__ = dict.__getitem__

        def to(self, _dev):
            return self

    class Tok:
        padding_side, pad_token, pad_token_id, eos_token = "right", None, 0, "</s>"

        def __call__(self, chunk, **kw):
            return Enc(input_ids=torch.zeros(len(chunk), 3, dtype=torch.long))

        def decode(self, ids, **_):
            return "x" * len(ids)

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, input_ids, **kw):
            seen.update(kw)
            return torch.cat([input_ids, torch.tensor([[5, 9]] * len(input_ids))], dim=1)

    out = E.generate_greedy(Model(), Tok(), ["a", "b"], max_new_tokens=4, batch_size=2)
    assert [o["finish_reason"] for o in out] == ["stop", "stop"]
    assert seen["repetition_penalty"] == 1.0 and seen["do_sample"] is False
