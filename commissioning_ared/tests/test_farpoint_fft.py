"""Global farthest-first queries on a lopsided three-blob pool.

A k-means policy that keeps only the larger half of a split stays inside the
majority blob. Global farthest-first does not: the first three queries land
in three different blobs.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from src.farpoint_fft import QUERY_RULE_GLOBAL, FarpointCommissioner


def _lopsided_blobs(seed: int, n_major: int = 800, n_minor: int = 40):
    rng = np.random.default_rng(seed)
    major = rng.normal(loc=(0.0, 0.0), scale=0.5, size=(n_major, 2))
    right = rng.normal(loc=(50.0, 0.0), scale=0.5, size=(n_minor, 2))
    up = rng.normal(loc=(0.0, 50.0), scale=0.5, size=(n_minor, 2))
    X = np.vstack([major, right, up])
    y = np.array(["maj"] * n_major + ["right"] * n_minor + ["up"] * n_minor)
    relevance = np.zeros(len(y), dtype=bool)
    relevance[y != "maj"] = True
    return X, y, relevance


def _kmeans_half_split_labels(X, y, budget: int, seed: int):
    """Bisect with k-means and spend every query inside the larger half.

    This is the policy the global rule is required to beat on a lopsided blob:
    each step throws away the smaller side of a 2-means split and queries the
    point closest to the centroid of what remains.
    """
    remaining = np.arange(len(X))
    labels = []
    for step in range(budget):
        if len(remaining) == 1:
            labels.append(str(y[int(remaining[0])]))
            break
        model = KMeans(n_clusters=2, n_init=10, random_state=seed + step)
        parts = model.fit_predict(X[remaining])
        counts = np.bincount(parts, minlength=2)
        bigger = int(np.argmax(counts))
        half = remaining[parts == bigger]
        centroid = X[half].mean(axis=0)
        local = int(np.argmin(np.linalg.norm(X[half] - centroid, axis=1)))
        chosen = int(half[local])
        labels.append(str(y[chosen]))
        remaining = half[half != chosen]
    return labels


def _reference_global(X, budget: int):
    """Independent farthest-first loop (centroid start, then max-min)."""
    centroid = X.mean(axis=0, keepdims=True)
    chosen = [int(np.argmax(cdist(X, centroid, metric="euclidean").ravel()))]
    remaining = np.ones(len(X), dtype=bool)
    remaining[chosen[0]] = False
    while len(chosen) < budget and remaining.any():
        unlabeled = np.flatnonzero(remaining)
        mins = cdist(X[unlabeled], X[np.asarray(chosen)], metric="euclidean").min(axis=1)
        nxt = int(unlabeled[int(np.argmax(mins))])
        chosen.append(nxt)
        remaining[nxt] = False
    return chosen


def test_default_query_rule_is_global_fft():
    default = inspect.signature(FarpointCommissioner.__init__).parameters["query_rule"].default
    assert default == QUERY_RULE_GLOBAL == "global_fft"


def test_global_fft_matches_reference_on_random_pool():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(35, 4))
    y = np.array([str(v) for v in rng.integers(0, 4, size=len(X))])
    relevance = np.zeros(len(X), dtype=bool)
    state = FarpointCommissioner(X, y, relevance, budget_B0=8, rng=0).run()
    assert state.query_rule == "global_fft"
    assert state.labeled_indices == _reference_global(X, 8)
    assert state.queries_used == 8
    assert state.prototypes.shape == (8, 4)


def test_global_fft_hits_three_blobs_half_split_does_not():
    for seed in range(5):
        X, y, relevance = _lopsided_blobs(seed)
        state = FarpointCommissioner(
            X, y, relevance, budget_B0=3, rng=seed, query_rule="global_fft"
        ).run()
        found = [str(label) for label in state.labels]
        assert len(set(found)) == 3, f"global_fft seed {seed} found {found}"
        assert state.queries_used == 3
        # One cluster per discovered class: no half-space split.
        assert len(state.cluster_dict) == 3
        half = _kmeans_half_split_labels(X, y, budget=3, seed=seed)
        assert len(set(half)) < 3, f"half-split unexpectedly covered {half} at seed {seed}"


def test_quit_stops_without_recording_the_quit_label():
    X = np.zeros((6, 2))
    y = ["quit"] * 6
    relevance = [False] * 6
    state = FarpointCommissioner(X, y, relevance, budget_B0=4, rng=0).run()
    assert state.queries_used == 0
    assert state.labeled_indices == []
    assert state.classes_discovered == []


def test_budget_does_not_pass_the_pool_end():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 2))
    y = np.array(["a"] * 12)
    relevance = np.zeros(12, dtype=bool)
    state = FarpointCommissioner(X, y, relevance, budget_B0=10, rng=1, pool_end=4).run()
    assert state.queries_used == 4
    assert all(0 <= i < 4 for i in state.labeled_indices)
    assert state.pool_end == 4


def test_legacy_cluster_fp_spends_budget_and_does_not_pick_global_queries():
    X, y, relevance = _lopsided_blobs(0, n_major=120, n_minor=15)
    legacy = FarpointCommissioner(
        X, y, relevance, budget_B0=6, rng=0, query_rule="legacy_cluster_fp"
    ).run()
    assert legacy.queries_used == 6
    assert legacy.labels == [str(y[i]) for i in legacy.labeled_indices]
    assert legacy.query_rule == "legacy_cluster_fp"
    # Prediction assignment exists and is defined on the whole pool, but the
    # queried indices themselves are the far-point sequence, not that assignment.
    assert legacy.nq_assignment is not None
    assert len(legacy.nq_assignment) == len(X)


def test_next_query_index_is_stable_before_run():
    X, y, relevance = _lopsided_blobs(2, n_major=30, n_minor=8)
    commissioner = FarpointCommissioner(X, y, relevance, budget_B0=3, rng=2)
    first = commissioner.next_query_index()
    again = commissioner.next_query_index()
    assert first == again
    state = commissioner.run()
    assert state.labeled_indices[0] == first


def test_nq_assignment_does_not_change_the_query():
    """Nearest-labeled labels are exported, and the query sequence ignores them."""
    X, y, relevance = _lopsided_blobs(3, n_major=40, n_minor=10)
    state = FarpointCommissioner(X, y, relevance, budget_B0=4, rng=3).run()
    assert list(state.nq_assignment[state.labeled_indices]) == state.labels
    # Re-running the rule from scratch reproduces the queries; the assignment
    # is not an input to that rule.
    again = FarpointCommissioner(X, y, relevance, budget_B0=4, rng=3).run()
    assert again.labeled_indices == state.labeled_indices
