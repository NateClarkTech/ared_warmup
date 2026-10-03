"""Seeded A/RED keeps the commissioning labels and comparison distances."""

from __future__ import annotations

import numpy as np

from src.datasets import DatasetBundle, relevance_from_labels
from src.farpoint_fft import commission_from_indices
from src.handoff import ARED, make_oracle, select_seed_items
from src.metrics import query_precision, relevant_recall_and_counts
from src.plotting import plot_all
from src.protocols import run_configured


def _state_two_classes():
    X = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 20.0],
            [0.4, 0.1],
            [50.0, 50.0],
        ],
        dtype=np.float64,
    )
    labels = ["a", "a", "b", "a", "c"]
    relevance = [True, True, False, True, False]
    # Query order: two of class a (so comp_distance is defined), then b, then c.
    state = commission_from_indices(X, labels, relevance, [0, 1, 2, 4], pool_end=5)
    return X, labels, relevance, state


def test_seed_selection_keeps_relevant_classes_first():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.2, 0.1]], dtype=np.float64)
    labels = ["a", "a", "b", "c", "a"]
    relevance = [True, True, True, False, True]
    state = commission_from_indices(X, labels, relevance, [0, 1, 2, 3, 4], pool_end=5)
    # Buffer holds 2 points. Relevant classes are a and b (c is irrelevant).
    # Both relevant classes must survive; the irrelevant point must not.
    chosen = select_seed_items(state, l_buf_size=2)
    chosen_labels = {item["label"] for item in chosen}
    assert "c" not in chosen_labels
    assert chosen_labels == {"a", "b"}


def test_seeded_ared_has_commissioning_labels_and_comp_distance():
    X, labels, relevance, state = _state_two_classes()
    assert "a" in state.known_relevant_labels
    assert state.queries_used == 4
    oracle = make_oracle(X, labels, relevance)
    ared = ARED.from_commissioning_state(
        state,
        oracle,
        kappa=0.5,
        l_buf_size=10,
        t_index=5,
        QS_VAR=1,
        K_COMP_PTS=2,
        SMART_FORGETTING_VAR=(0, 0.01),
        VERBOSE_FLAGS=[],
    )
    assert ared.num_pts_streamed == 5
    assert ared.abs_index == 4
    assert ared.num_queries == 0
    stored_idx = ared.l_buf.true_abs_idx_circular_buffer.get_array()
    stored_labels = ared.l_buf.label_circular_buffer.get_array()
    assert set(stored_idx) == set(state.labeled_indices)
    assert set(stored_labels) == set(state.labels)
    assert ared.subspace_partition.set_of_known_labels == set(state.known_labels)
    distances = {
        cluster.label: cluster.comp_distance
        for cluster in ared.subspace_partition.cluster_dict.values()
    }
    # Class a was seeded with (0, 0) and (3, 0). Average nearest-neighbor
    # distance of that pair is 3. Singletons stay at 0.
    assert abs(distances["a"] - 3.0) < 1e-6
    assert distances["b"] == 0.0
    assert distances["c"] == 0.0
    for cluster in ared.subspace_partition.cluster_dict.values():
        if cluster.label == "a":
            assert cluster.relevance is True
        if cluster.label == "b":
            assert cluster.relevance is False


def test_comp_distance_is_close():
    X, labels, relevance, state = _state_two_classes()
    oracle = make_oracle(X, labels, relevance)
    ared = ARED.from_commissioning_state(
        state,
        oracle,
        kappa=0.5,
        l_buf_size=10,
        t_index=5,
        QS_VAR=1,
        SMART_FORGETTING_VAR=(0, 0.01),
    )
    for cluster in ared.subspace_partition.cluster_dict.values():
        if cluster.label == "a":
            assert abs(cluster.comp_distance - 3.0) < 1e-6


def test_relevant_recall_uses_vendored_diagonal_ratio():
    labels = ["a", "a", "b", "b", "b"]
    relevance = [False, False, True, True, True]
    X = np.zeros((5, 2))
    oracle = make_oracle(X, labels, relevance)
    # Rows/cols follow the sorted label map: "a" then "b".
    assert list(oracle.int_str_label_bidict.keys()) == ["a", "b"]
    cm = np.zeros((2, 2), dtype=float)
    cm[1, 1] = 2  # queried relevant
    cm[1, 0] = 3  # streamed relevant, not queried
    ared = type("Obj", (), {"oracle": oracle})()
    recall, queried, streamed = relevant_recall_and_counts(cm, ["b"], ared)
    assert streamed == 5
    assert queried == 2
    assert abs(recall - 0.4) < 1e-9
    assert query_precision(2, 8) == 0.25
    assert query_precision(0, 0) == 0.0


def _toy_bundle():
    rng = np.random.default_rng(0)
    blobs = [
        rng.normal(loc=(0.0, 0.0), scale=0.35, size=(50, 2)),
        rng.normal(loc=(12.0, 0.0), scale=0.35, size=(18, 2)),
        rng.normal(loc=(0.0, 12.0), scale=0.35, size=(10, 2)),
    ]
    X = np.vstack(blobs)
    raw = np.array(["0"] * 50 + ["1"] * 18 + ["2"] * 10)
    labels, relevance, rel_classes, sparsity = relevance_from_labels(raw, n_rel_classes=2)
    return DatasetBundle("TOY", X, labels, relevance, rel_classes, sparsity)


def test_cold_and_farpoint_handoff_on_a_toy_stream():
    bundle = _toy_bundle()
    common = {
        "data": "TOY",
        "bundle": bundle,
        "t_frac": 0.34,
        "B0": 6,
        "kappa": 0.25,
        "l_buf_size": 200,
        "n_rel_classes": 2,
        "seed": 0,
        "stream": "stationary",
        "qs_var": 1,
        "k_comp_pts": 2,
        "smart_forgetting": (0, 0.01),
        "verbose": False,
        "progress": False,
        "query_rule": "global_fft",
    }
    cold = run_configured({**common, "method": "cold_ared"})
    far = run_configured({**common, "method": "farpoint_then_ared"})
    for record in (cold, far):
        watch = record["watch"]
        assert 0.0 <= watch["query_precision"] <= 1.0
        assert 0.0 <= watch["relevant_recall"] <= 1.0
        assert record["N"] == len(bundle.X)
        assert record["watch"]["num_queries"] >= 0
    assert far["commissioning"]["queries_used"] == 6
    assert far["commissioning"]["n_classes_discovered"] == 3
    assert set(far["commissioning"]["relevant_classes_discovered"]).issubset(set(bundle.rel_classes))
    # Watch phase starts at T: discovery of a class found in the pool is < T.
    T = far["T"]
    for label in far["commissioning"]["classes_discovered"]:
        latency = far["watch"]["discovery_latency"].get(label)
        if label in bundle.rel_classes and latency is not None:
            assert latency < T


def test_figures_render_from_records(tmp_path):
    def record(method, stream, b0, kappa, rr, qp):
        return {
            "method": method,
            "data": "NICE",
            "stream": stream,
            "seed": 0,
            "t_frac": 0.2,
            "T": 20,
            "N": 100,
            "B0": b0,
            "kappa": kappa,
            "l_buf_size": 50,
            "n_rel_classes": 2,
            "query_rule": "global_fft",
            "held_out_class": "b" if stream == "late_arrival" else None,
            "n_rel_classes_in_pool": 1,
            "post_T_rel_mass_from_pool_classes": 0.8,
            "seconds": 0.1,
            "rel_classes": ["a", "b"],
            "commissioning": {
                "queries_used": 0 if method == "cold_ared" else b0,
                "classes_discovered": ["a"],
                "relevant_classes_discovered": ["a"],
                "n_classes_discovered": 1,
                "n_relevant_discovered": 1,
            },
            "watch": {
                "query_precision": qp,
                "relevant_recall": rr,
                "num_queries": 5,
                "num_correct_queries": 2,
                "query_rate": 0.05,
                "relevant_queried": 2,
                "relevant_streamed": 4,
                "missed_relevant_mass": 2,
                "missed_relevant_fraction": 0.5,
                "discovery_latency": {"a": 3, "b": 40 if stream == "late_arrival" else 7},
                "queries_on_known_irrelevant": 1,
            },
            "whole": {
                "total_labels": 8,
                "relevant_examples_found": 2,
                "labels_per_relevant_found": 4.0,
            },
        }

    records = []
    for method, rr, qp in (
        ("cold_ared", 0.2, 0.4),
        ("random_then_ared", 0.3, 0.5),
        ("farpoint_then_ared", 0.6, 0.7),
        ("oracle_then_ared", 0.8, 0.75),
    ):
        for stream in ("stationary", "late_arrival"):
            for b0, kappa in ((10, 0.5), (20, 1.0)):
                records.append(record(method, stream, b0, kappa, rr, qp))
    written = plot_all(records, tmp_path)
    names = {path.name for path in written}
    assert "fig1_watch_rr_vs_b0_stationary.png" in names
    assert "fig2_watch_rr_vs_b0_late_arrival.png" in names
    assert "fig3_qp_vs_rr_kappa.png" in names
    assert "fig4_discovery_latency.png" in names
