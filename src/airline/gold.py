"""Wrapper around RuleArena airline gold computation.

Handles path resolution so we can call compute_answer from any working directory.
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd


_RULEARENA_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "rulearena"
_AIRLINE_DIR = _RULEARENA_ROOT / "airline"


def load_checking_fee_tables() -> list[dict]:
    """Load the 4 bag fee CSV tables (bag_1..bag_4, direction 0 and 1)."""
    tables = []
    for bag_num in range(1, 5):
        us_departure = pd.read_csv(
            _AIRLINE_DIR / "fee_tables" / f"bag_{bag_num}" / "0.csv", index_col=0)
        us_arrival = pd.read_csv(
            _AIRLINE_DIR / "fee_tables" / f"bag_{bag_num}" / "1.csv", index_col=0)
        tables.append({0: us_departure, 1: us_arrival})
    return tables


_CHECK_BASE_TABLES = None


def get_check_base_tables():
    global _CHECK_BASE_TABLES
    if _CHECK_BASE_TABLES is None:
        _CHECK_BASE_TABLES = load_checking_fee_tables()
    return _CHECK_BASE_TABLES


# Re-implement compute_answer locally so we don't depend on cwd
COMPLEMENTARY_FIRST = [
    "China", "Hong Kong", "Japan", "South Korea", "India",
    "Qatar", "Haiti", "Cuba", "Panama", "Colombia",
    "Ecuador", "Peru", "South America", "Israel"
]


def compute_oversize(bag: dict, routine: str) -> int:
    total_size = sum(bag["size"])
    if total_size <= 62:
        return 0
    if total_size <= 65:
        return 30
    if routine in ["Panama", "South America", "Peru", "Colombia", "Ecuador",
                    "Europe", "Israel", "Qatar"]:
        return 150
    return 200


def compute_overweight(bag: dict, routine: str, customer_class: str,
                       complementary: bool) -> int:
    w = bag["weight"]
    if routine in ["Australia", "New Zealand"]:
        if complementary:
            return 0 if w <= 70 else 200
        if w <= 50:
            return 0
        if w <= 53:
            return 30
        if w <= 70:
            return 200 if routine == "Cuba" else 100
        if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        return 200
    if complementary and customer_class in ["Business", "First"]:
        if w <= 70:
            return 0
        if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
            return 450
        return 200
    if w <= 50:
        return 0
    if w <= 53:
        return 30
    if w <= 70:
        return 200 if routine == "Cuba" else 100
    if routine in ["India", "China", "Japan", "South Korea", "Hong Kong"]:
        return 450
    return 200


def compute_base(bag_list, direction, routine, customer_class, tables):
    check_base = []
    for bag_id, _ in enumerate(bag_list):
        idx = min(3, bag_id)
        check_base.append(tables[idx][direction][customer_class][routine])
    return check_base


def compute_airline_gold(info_dict: dict) -> int:
    """Compute the gold total cost for an airline problem.

    Args:
        info_dict: dict with keys base_price, customer_class, routine,
                   direction, bag_list

    Returns:
        total_cost (int)
    """
    tables = get_check_base_tables()
    bag_list = info_dict["bag_list"][1:]  # skip carry-on

    # Greedy reordering for complementary benefit
    oversize_cost = [compute_oversize(b, info_dict["routine"]) for b in bag_list]
    ow_comp = [compute_overweight(b, info_dict["routine"], info_dict["customer_class"], True)
               for b in bag_list]
    ow_nocomp = [compute_overweight(b, info_dict["routine"], info_dict["customer_class"], False)
                 for b in bag_list]
    violation_comp = np.maximum(oversize_cost, ow_comp)
    violation_nocomp = np.maximum(oversize_cost, ow_nocomp)
    gain = np.array(violation_nocomp) - np.array(violation_comp)
    order = np.argsort(-gain)
    bag_list_sorted = [bag_list[i] for i in order]

    check_base = compute_base(bag_list_sorted, info_dict["direction"],
                              info_dict["routine"], info_dict["customer_class"],
                              tables)
    complementary = [(x == 0) for x in check_base]
    os_cost = [compute_oversize(b, info_dict["routine"]) for b in bag_list_sorted]
    ow_cost = [compute_overweight(b, info_dict["routine"], info_dict["customer_class"], c)
               for b, c in zip(bag_list_sorted, complementary)]
    violation_cost = np.maximum(os_cost, ow_cost).sum()
    total_check = int(np.sum(check_base) + violation_cost)

    return info_dict["base_price"] + total_check


def load_airline_problems(complexity: int = 0) -> list[dict]:
    """Load synthesized airline problems from RuleArena."""
    path = _AIRLINE_DIR / "synthesized_problems" / f"comp_{complexity}.jsonl"
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems
