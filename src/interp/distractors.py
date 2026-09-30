"""Irrelevant-information perturbations for RuleArena airline prompts.

Two insertion sites:
  - "prompt":    a sentence inserted in the middle of the passenger's item
                 list, before the model reads the question.
  - "reasoning": a sentence inserted in the middle of an arithmetic step of
                 the model's own clean reasoning; the model then continues
                 from there (prefill).

None of the sentences carries information about bags, fares or classes, so
the gold answer is unchanged by construction. A model that is robust should
give the same answer with and without them.
"""

from __future__ import annotations

import re

# Default distractors (the prompts are in English, so the sentences are too).
# Keys are the Portuguese originals they translate.
DEFAULT_DISTRACTORS = {
    "estava_contando_1+1": "I was counting 1+1.",
    "parei_para_respirar": "I stopped to breathe.",
    "qual_o_valor": "What is the value?",
}

_ITEM_LINE_RE = re.compile(r"^\d+\. .*$", re.MULTILINE)
# Optional markdown bold between "is" and "$": some models (Qwen3-14B) write
# "The total cost is **$1,434**.", which the evaluator already accepts
_ANSWER_RE = re.compile(r"The total cost is \*{0,2}\$([0-9][0-9,]*(?:\.[0-9]+)?)")
# "+ " between two amounts, e.g. "$190 + $195": the model is mid-sum
_PLUS_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:\s*\([^)]*\))?\s*\+\s")
_EQUALS_RE = re.compile(r"\s=\s")


def insert_in_question(question: str, sentence: str) -> str:
    """Insert `sentence` as its own line after the middle item of the list."""
    items = list(_ITEM_LINE_RE.finditer(question))
    if not items:
        raise ValueError("no numbered item list found in question")
    anchor = items[(len(items) - 1) // 2]
    return question[:anchor.end()] + "\n" + sentence + question[anchor.end():]


def find_reasoning_cut(reasoning: str, min_frac: float = 0.4) -> tuple[int, str]:
    """Character offset in the middle of a calculation where text can be spliced.

    Prefers the first "+ " of a sum that starts after `min_frac` of the text,
    then the first " = ", then the first newline after `min_frac`. The answer
    sentence itself is never used as a cut point.
    """
    answer = _ANSWER_RE.search(reasoning)
    limit = answer.start() if answer else len(reasoning)
    start = int(min_frac * limit)
    for kind, pattern in (("plus", _PLUS_RE), ("equals", _EQUALS_RE)):
        for m in pattern.finditer(reasoning, start, limit):
            return m.end(), kind
    nl = reasoning.find("\n", start, limit)
    if nl != -1:
        return nl + 1, "newline"
    raise ValueError("no cut point found before the final answer")


def insert_in_reasoning(reasoning_prefix: str, sentence: str) -> str:
    """Append `sentence` to a reasoning prefix that ends mid-calculation."""
    sep = "" if reasoning_prefix.endswith((" ", "\n")) else " "
    return f"{reasoning_prefix}{sep}{sentence} "


def split_final_answer(response: str) -> tuple[str, str] | None:
    """Split a response at its last "The total cost is $X".

    Returns (text up to and including "$", answer string as printed, e.g.
    "1,365"), or None if the answer sentence is missing. Markdown bold is
    allowed ("is **$1,365**."): the prefix then ends in "**$".
    """
    matches = list(_ANSWER_RE.finditer(response))
    if not matches:
        return None
    m = matches[-1]
    return response[:m.start(1)], m.group(1)
