"""Pool length, farthest-first order, comparison distance, and forgetting."""

from __future__ import annotations

import numpy as np

from ared.algorithm import run
from ared.config import load_config
from ared.detector import ClusterStore, Detector, Hit, Memory
from ared.stream import synthetic_stream
from comparison_distance import build as build_comparison
from comparison_distance.average_nearest_neighbor import AverageNearestNeighbor
from comparison_distance.diameter import Diameter
from neighborhood_merge import build as build_neighborhood
from singleton_merge import build as build_singleton


def reference_far_points(prefix) -> list[int]:
    """Centroid, then max-min, written as plain loops rather than the program."""
    points = np.asarray(prefix, dtype=np.float64)
    count = len(points)
    centroid = points.mean(axis=0)
    first = max(range(count), key=lambda index: float(np.linalg.norm(points[index] - centroid)))
    chosen = [first]
    remaining = [index for index in range(count) if index != first]
    while remaining:
        best_index = None
        best_distance = None
        for index in remaining:
            nearest = min(float(np.linalg.norm(points[index] - points[chosen_index])) for chosen_index in chosen)
            if best_distance is None or nearest > best_distance:
                best_distance = nearest
                best_index = index
        chosen.append(best_index)
        remaining.remove(best_index)
    return chosen


def _labels_for(count: int):
    labels = [f"c{index % 4}" for index in range(count)]
    relevance = [(index % 5) == 0 for index in range(count)]
    return labels, relevance


def test_warmup_from_json_is_the_far_point_prefix():
    rng = np.random.default_rng(11)
    points = rng.normal(size=(20, 3))
    labels, relevance = _labels_for(len(points))

    short_cfg = load_config("warmup_4.json")
    long_cfg = load_config("warmup_8.json")
    assert short_cfg.warmup != long_cfg.warmup

    short = run("warmup_4.json", points=points, labels=labels, relevance=relevance)
    long = run("warmup_8.json", points=points, labels=labels, relevance=relevance)

    assert short.pool_length == short_cfg.warmup
    assert long.pool_length == long_cfg.warmup
    assert short.far_point_indices == reference_far_points(points[: short.pool_length])
    assert long.far_point_indices == reference_far_points(points[: long.pool_length])
    assert len(short.far_point_indices) <= short.pool_length
    assert len(long.far_point_indices) <= long.pool_length
    assert all(0 <= index < short.pool_length for index in short.far_point_indices)
    assert all(0 <= index < long.pool_length for index in long.far_point_indices)
    assert short.pool_length != long.pool_length
    assert isinstance(short.stream_query_count, int)
    assert isinstance(long.stream_query_count, int)


def test_builtin_stream_returns_pool_far_points_and_stream_queries():
    config = load_config("tiny.json")
    result = run("tiny.json")
    assert result.config_name == "tiny.json"
    assert result.pool_length == config.warmup
    generated, _labels, _relevance = synthetic_stream(config.stream_seed, config.stream_n, config.stream_dim)
    assert result.far_point_indices == reference_far_points(generated[: config.warmup])
    assert len(result.far_point_indices) <= result.pool_length
    assert result.stream_query_count >= 0


def _line_stream():
    # 0, 1, and 10 in the pool. 15 is streamed. Diameter is 10; average nearest neighbor is 11/3.
    points = np.array([[0.0], [1.0], [10.0], [15.0]], dtype=np.float64)
    labels = ["a", "a", "a", "a"]
    relevance = [False, False, False, False]
    return points, labels, relevance


def test_comparison_distance_strategies_disagree_and_the_config_is_what_the_detector_uses():
    points, labels, relevance = _line_stream()
    pool = points[:3]
    diameter = Diameter().measure(pool)
    neighbor = AverageNearestNeighbor().measure(pool)
    assert diameter != neighbor

    distance = min(float(np.linalg.norm(points[3] - points[index])) for index in range(3))
    diameter_cfg = load_config("tiny.json")
    neighbor_cfg = load_config("neighbor_tiny.json")
    assert diameter_cfg.comparison_distance == "diameter"
    assert neighbor_cfg.comparison_distance == "average_nearest_neighbor"
    assert diameter_cfg.warmup == neighbor_cfg.warmup == 3
    assert diameter_cfg.kappa == neighbor_cfg.kappa
    assert diameter_cfg.buffer_size == neighbor_cfg.buffer_size

    diameter_run = run("tiny.json", points=points, labels=labels, relevance=relevance)
    neighbor_run = run("neighbor_tiny.json", points=points, labels=labels, relevance=relevance)
    assert diameter_run.pool_length == diameter_cfg.warmup
    assert neighbor_run.pool_length == neighbor_cfg.warmup

    for config, result in ((diameter_cfg, diameter_run), (neighbor_cfg, neighbor_run)):
        measure = build_comparison(config.comparison_distance).measure(pool)
        expects_query = distance * config.kappa > measure
        assert (result.stream_query_count == 1) is expects_query

    assert diameter_run.stream_query_count != neighbor_run.stream_query_count


def test_nearby_relevant_point_forces_a_query_inside_the_comparison_distance():
    config = load_config("tiny.json")
    points = np.array([[0.0], [1.0], [1.2]], dtype=np.float64)
    labels = ["a", "a", "b"]

    forced = Detector.from_config(config, points, labels, [True, False, False])
    forced.remember(points[0], "a", True)
    forced.remember(points[1], "a", False)
    assert forced.stream_point(2) is True

    quiet = Detector.from_config(config, points, labels, [False, False, False])
    quiet.remember(points[0], "a", False)
    quiet.remember(points[1], "a", False)
    assert quiet.stream_point(2) is False


def test_keep_relevant_retains_a_label_plain_forgetting_drops():
    plain_cfg = load_config("forget_plain.json")
    keep_cfg = load_config("forget_keep_relevant.json")
    assert plain_cfg.buffer_size == keep_cfg.buffer_size
    assert plain_cfg.buffer_size < 3
    assert plain_cfg.smart_forgetting == "plain"
    assert keep_cfg.smart_forgetting == "keep_relevant"

    vectors = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
    labels = ["kept", "gone", "other"]
    relevance = [True, False, False]
    points = np.vstack(vectors)

    def retained(config):
        detector = Detector.from_config(config, points, labels, relevance)
        for vector, label, relevant in zip(vectors, labels, relevance):
            detector.remember(vector, label, relevant)
        return detector.buffered_labels()

    plain_labels = retained(plain_cfg)
    keep_labels = retained(keep_cfg)
    assert "kept" not in plain_labels
    assert "kept" in keep_labels


def test_same_label_neighborhood_merges_and_disabled_does_not():
    def store_with_two_clusters():
        store = ClusterStore(Diameter())
        first = store.open("a", False)
        second = store.open("a", False)
        left = Memory(np.array([0.0, 0.0]), "a", False, first.id)
        right = Memory(np.array([1.0, 0.0]), "a", False, second.id)
        store.add_member(first, left)
        store.add_member(second, right)
        return store, [Hit(left, 0.1), Hit(right, 0.2)]

    left_alone, left_hits = store_with_two_clusters()
    build_neighborhood("disabled").before_query(left_alone, left_hits)
    assert len(left_alone.all()) == 2

    merged, merged_hits = store_with_two_clusters()
    build_neighborhood("same_label").before_query(merged, merged_hits)
    assert len(merged.all()) == 1
    assert merged_hits[0].cluster_id == merged_hits[1].cluster_id


def test_singleton_merge_absorbs_a_one_point_cluster_and_disabled_does_not():
    def store_with_a_singleton():
        store = ClusterStore(Diameter())
        group = store.open("a", False)
        alone = store.open("a", False)
        store.add_member(group, Memory(np.array([0.0]), "a", False, group.id))
        store.add_member(group, Memory(np.array([0.2]), "a", False, group.id))
        store.add_member(alone, Memory(np.array([5.0]), "a", False, alone.id))
        return store

    untouched = store_with_a_singleton()
    build_singleton("disabled").apply(untouched)
    assert len(untouched.all()) == 2

    folded = store_with_a_singleton()
    build_singleton("merge_singletons").apply(folded)
    assert len(folded.all()) == 1
    assert len(folded.all()[0].members) == 3
