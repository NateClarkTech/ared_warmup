"""Stream constructions are permutations, not new draws."""

from __future__ import annotations

import numpy as np

from src.protocols import _oracle_indices
from src.streams import (
    assert_late_arrival,
    assert_stationary,
    late_arrival_permutation,
    repair_stationary,
    stationary_permutation,
)


def test_repair_puts_every_repeated_class_on_both_sides():
    labels = np.array(["a", "a", "a", "a", "b", "b", "c", "c"], dtype=object)
    perm = np.arange(len(labels))
    repaired = repair_stationary(perm, labels[perm], T=4)
    assert sorted(repaired.tolist()) == list(range(len(labels)))
    assert_stationary(labels[repaired], 4)


def test_stationary_permutation_holds_over_seeds():
    labels = np.array(["a"] * 30 + ["b"] * 12 + ["c"] * 6 + ["d"] * 4, dtype=object)
    T = 15
    for seed in range(6):
        perm = stationary_permutation(labels, T, np.random.default_rng(seed))
        assert sorted(perm.tolist()) == list(range(len(labels)))
        assert_stationary(labels[perm], T)


def test_late_arrival_holds_out_the_rarest_relevant_class():
    labels = np.array(["a"] * 40 + ["b"] * 8 + ["c"] * 3, dtype=object)
    perm, held = late_arrival_permutation(labels, ["b", "c"], T=12, rng=np.random.default_rng(0))
    assert held == "c"
    assert sorted(perm.tolist()) == list(range(len(labels)))
    assert_late_arrival(labels[perm], 12, held)


def test_oracle_budget_covers_rarest_classes_first():
    labels = ["freq"] * 10 + ["mid"] * 3 + ["rare"]
    picks = _oracle_indices(labels, T=len(labels), budget=3, rng=np.random.default_rng(0))
    assert [labels[i] for i in picks[:3]] == ["rare", "mid", "freq"]
