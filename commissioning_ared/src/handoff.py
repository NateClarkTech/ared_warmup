"""Seed a streaming A/RED model from a commissioning state.

The detector's query test (distance * kappa > comp_distance) is not modified.
Labeled pool points are inserted into the finite buffer and one detector
cluster is created per commissioning cluster. Comparison distances are
recomputed from those seeded points with the same QS_VAR the watch phase
will use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from bidict import bidict

from src.farpoint_fft import CommissioningState
from src.vendor_bootstrap import import_ared_symbols

ARED, Oracle, _calculate_single_rel_recall = import_ared_symbols()


class StreamOracle(Oracle):
    """Vendor oracle with plain ``str`` keys in the label map.

    The vendored map is built with ``np.unique(..., dtype=str)``. Subclassing
    keeps those keys as Python strings so later lookups from query results match.
    """

    def create_label_bidict(self, labels):
        flat = np.asarray(labels, dtype=object).reshape(-1)
        unique = sorted({str(value) for value in flat.tolist()})
        return bidict({label: idx for idx, label in enumerate(unique)})

# Defaults match the streaming driver's checked-in parking/NICE settings
# where a value is unambiguous: k = 2, average nearest-neighbor comparison
# distance, no image augmentation, class-fraction forgetting at 1%.
DEFAULT_K_COMP_PTS = 2
DEFAULT_QS_VAR = 1
DEFAULT_DATA_AUG_VAR = (0, (0,))
DEFAULT_NGHBHOOD_MERGE = False
DEFAULT_SINGLETON_MERGE = False
DEFAULT_SMART_FORGETTING_VAR = (3, 0.01)
DEFAULT_VERBOSE_FLAGS: Sequence[int] = ()


def _diverse_subset(items: List[Dict[str, Any]], k: int, prototypes: np.ndarray) -> List[Dict[str, Any]]:
    """Farthest-first subset. The most recent item is the seed (recency bias)."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    if len(prototypes) != len(items):
        raise ValueError("prototypes must align with items")
    chosen_pos = [len(items) - 1]
    chosen = {chosen_pos[0]}
    while len(chosen_pos) < k:
        selected = prototypes[[p for p in chosen_pos]]
        rest = [p for p in range(len(items)) if p not in chosen]
        dists = cdist(prototypes[rest], selected, metric="euclidean").min(axis=1)
        # Tiny bonus so an exact tie prefers the later (more recent) point.
        bonus = np.asarray(rest, dtype=np.float64) * 1e-12
        pick = rest[int(np.argmax(dists + bonus))]
        chosen.add(pick)
        chosen_pos.append(pick)
    return [items[p] for p in sorted(chosen_pos)]


def _cover_classes(items: List[Dict[str, Any]], k: int, prototypes: np.ndarray) -> List[Dict[str, Any]]:
    """Keep one earliest prototype per class (rarest class first), then diversify."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    by_class: Dict[Any, List[int]] = {}
    for pos, item in enumerate(items):
        by_class.setdefault(item["label"], []).append(pos)
    class_order = sorted(by_class, key=lambda label: (len(by_class[label]), str(label)))
    seed_pos = [by_class[label][0] for label in class_order[:k]]
    if len(seed_pos) >= k:
        return [items[p] for p in seed_pos[:k]]
    remaining_pos = [p for p in range(len(items)) if p not in set(seed_pos)]
    extra = _diverse_subset(
        [items[p] for p in remaining_pos],
        k - len(seed_pos),
        prototypes[remaining_pos] if remaining_pos else prototypes[:0],
    )
    # Map extra items back by identity.
    extra_ids = {id(item) for item in extra}
    extra_pos = [p for p in remaining_pos if id(items[p]) in extra_ids]
    keep = set(seed_pos + extra_pos)
    return [items[p] for p in range(len(items)) if p in keep]


def select_seed_items(state: CommissioningState, l_buf_size: int) -> List[Dict[str, Any]]:
    """Choose labeled points that fit in the finite buffer.

    Inclusion priority: every relevant-class prototype first, then a diverse
    / recent fill from the other labeled points. If the relevant prototypes
    themselves do not fit, keep one point from each relevant class (smallest
    classes first) and fill the rest by farthest-first among relevant points.

    The returned list is the inclusion set in query order. Callers that insert
    into a FIFO buffer should append non-relevant points before relevant ones
    so a full buffer drops non-relevant points first; ``seed_ared`` does that.
    """
    if l_buf_size <= 0:
        raise ValueError(f"l_buf_size must be positive, got {l_buf_size}")
    n = len(state.labeled_indices)
    if n == 0:
        return []
    if state.prototypes.shape[0] != n:
        raise ValueError(
            f"prototypes rows ({state.prototypes.shape[0]}) != labeled points ({n})"
        )
    items = [
        {
            "index": int(index),
            "label": label,
            "relevance": bool(rel),
            "row": int(row),
        }
        for row, (index, label, rel) in enumerate(
            zip(state.labeled_indices, state.labels, state.relevances)
        )
    ]
    if n <= l_buf_size:
        return items

    relevant_labels = set(state.known_relevant_labels)
    rel_pos = [i for i, item in enumerate(items) if item["label"] in relevant_labels]
    oth_pos = [i for i, item in enumerate(items) if item["label"] not in relevant_labels]
    rel_items = [items[i] for i in rel_pos]
    oth_items = [items[i] for i in oth_pos]
    rel_proto = state.prototypes[rel_pos] if rel_pos else state.prototypes[:0]
    oth_proto = state.prototypes[oth_pos] if oth_pos else state.prototypes[:0]

    if len(rel_items) >= l_buf_size:
        return _cover_classes(rel_items, l_buf_size, rel_proto)

    need = l_buf_size - len(rel_items)
    filler = _diverse_subset(oth_items, need, oth_proto)
    return rel_items + filler


def _append_order(items: List[Dict[str, Any]], relevant_labels: set) -> List[Dict[str, Any]]:
    """FIFO append order: non-relevant first, relevant-class points last (newest)."""
    others = [item for item in items if item["label"] not in relevant_labels]
    relevant = [item for item in items if item["label"] in relevant_labels]
    return others + relevant


def wait_for_buffer_trees(buffer, timeout: float = 30.0) -> None:
    """Block until a background ball-tree build started by inserts has finished."""
    import time

    deadline = time.time() + timeout
    while getattr(buffer, "_building_tree", False) and time.time() < deadline:
        time.sleep(0.01)


def make_pair_labels(labels: Sequence[Any], relevance: Sequence[Any]):
    """Oracle label table: list of ``(label_str, relevance_bool)``."""
    if len(labels) != len(relevance):
        raise ValueError("labels and relevance must have the same length")
    return [(str(label), bool(rel)) for label, rel in zip(labels, relevance)]


def make_oracle(X: np.ndarray, labels: Sequence[Any], relevance: Sequence[Any]):
    """Vendor oracle over an aligned stream. Indices are stream positions."""
    n = len(labels)
    if len(relevance) != n:
        raise ValueError("labels and relevance must have the same length")
    table = np.empty((n, 2), dtype=object)
    table[:, 0] = [str(value) for value in labels]
    table[:, 1] = [bool(value) for value in relevance]
    return StreamOracle(np.asarray(X), table)


def _ared_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "K_COMP_PTS": int(kwargs.get("K_COMP_PTS", DEFAULT_K_COMP_PTS)),
        "QS_VAR": int(kwargs.get("QS_VAR", DEFAULT_QS_VAR)),
        "DATA_AUG_VAR": tuple(kwargs.get("DATA_AUG_VAR", DEFAULT_DATA_AUG_VAR)),
        "NGHBHOOD_MERGE": bool(kwargs.get("NGHBHOOD_MERGE", DEFAULT_NGHBHOOD_MERGE)),
        "SINGLETON_MERGE": bool(kwargs.get("SINGLETON_MERGE", DEFAULT_SINGLETON_MERGE)),
        "SMART_FORGETTING_VAR": tuple(kwargs.get("SMART_FORGETTING_VAR", DEFAULT_SMART_FORGETTING_VAR)),
        "VERBOSE_FLAGS": list(kwargs.get("VERBOSE_FLAGS", DEFAULT_VERBOSE_FLAGS)),
    }


def from_commissioning_state(cls, state: CommissioningState, oracle, kappa, l_buf_size, **ared_kwargs):
    """Build an A/RED model whose clusters and buffer match ``state``.

    ``t_index`` (keyword, default ``state.pool_end``) is the first stream
    position the watch phase will pass to ``process_point``. Counters are set
    so that call reads oracle index ``t_index`` and stores absolute index
    ``t_index``. No comparison-distance formula is changed: each seeded
    cluster recomputes ``comp_distance`` with ``Cluster.merge_comp_distances``
    under the configured ``QS_VAR``.
    """
    if state is None:
        raise ValueError("commissioning state is required")
    opts = _ared_kwargs(ared_kwargs)
    t_index = ared_kwargs.get("t_index", None)
    if t_index is None:
        t_index = int(state.pool_end)
    t_index = int(t_index)
    if t_index < 0:
        raise ValueError(f"t_index must be non-negative, got {t_index}")

    ared = cls(
        oracle,
        float(kappa),
        int(l_buf_size),
        opts["K_COMP_PTS"],
        opts["QS_VAR"],
        opts["DATA_AUG_VAR"],
        opts["NGHBHOOD_MERGE"],
        opts["SINGLETON_MERGE"],
        opts["SMART_FORGETTING_VAR"],
        opts["VERBOSE_FLAGS"],
    )
    _seed_clusters_and_buffer(ared, state, int(l_buf_size), opts["QS_VAR"])
    # process_point / process_first_point increment both counters before use.
    ared.num_pts_streamed = t_index
    ared.abs_index = t_index - 1
    ared.num_queries = 0
    ared.num_correct_queries = 0
    ared.anom_only_queries = 0
    ared.rel_only_queries = 0
    ared.both_a_and_r_queries = 0
    ared.cumulative_relevant_seen = 0
    return ared


def _seed_clusters_and_buffer(ared, state: CommissioningState, l_buf_size: int, qs_var: int) -> None:
    chosen = select_seed_items(state, l_buf_size)
    if not chosen:
        return
    relevant_labels = set(state.known_relevant_labels)
    ordered = _append_order(chosen, relevant_labels)

    # Cluster keys follow creation order. Only clusters that still have at
    # least one buffered point are created (an empty cluster cannot answer
    # a nearest-neighbor query).
    by_source: Dict[int, List[Dict[str, Any]]] = {}
    for item in chosen:
        by_source.setdefault(_source_cluster_id(state, item["index"]), []).append(item)

    planned = []
    next_key = 0
    for source_id in sorted(by_source):
        members = by_source[source_id]
        info = state.cluster_dict[source_id]
        # Preserve labeled-point order from the commissioning cluster, restricted
        # to the points that actually fit in the buffer.
        keep = {item["index"] for item in members}
        l_pts = [int(i) for i in info["l_pt_idxs"] if int(i) in keep]
        if not l_pts:
            continue
        planned.append(
            {
                "key": next_key,
                "label": info["label"],
                "relevance": bool(info["relevance"]),
                "l_pts": l_pts,
            }
        )
        next_key += 1

    key_of_index = {}
    for plan in planned:
        for index in plan["l_pts"]:
            key_of_index[index] = plan["key"]

    proto_of = {
        int(index): np.asarray(state.prototypes[row], dtype=np.float64)
        for row, index in enumerate(state.labeled_indices)
    }
    for item in ordered:
        index = int(item["index"])
        if index not in key_of_index:
            continue
        ared.l_buf.insert_pt(
            proto_of[index],
            key_of_index[index],
            item["label"],
            bool(item["relevance"]),
            index,
        )
    wait_for_buffer_trees(ared.l_buf)

    for plan in planned:
        ared.subspace_partition.create_new_cluster(
            plan["label"],
            bool(plan["relevance"]),
            list(plan["l_pts"]),
            [],
            qs_var,
        )
        cluster = ared.subspace_partition.cluster_dict[plan["key"]]
        # create_new_cluster already computes comp_distance when the cluster
        # has two or more points. Recompute explicitly so a one-point cluster
        # stays at 0 and a larger cluster matches QS_VAR on the seeded points.
        cluster.merge_comp_distances(ared.l_buf, qs_var)


def _source_cluster_id(state: CommissioningState, index: int) -> int:
    for cid, info in state.cluster_dict.items():
        if int(index) in {int(i) for i in info["l_pt_idxs"]}:
            return int(cid)
    raise KeyError(f"labeled index {index} is not in any commissioning cluster")


# Public constructor requested on the detector class. Defined here so the
# vendored file stays unmodified.
ARED.from_commissioning_state = classmethod(from_commissioning_state)


def empty_ared(oracle, kappa: float, l_buf_size: int, **ared_kwargs):
    """Cold-start model. Counters remain at the vendor defaults (index -1 / 0)."""
    opts = _ared_kwargs(ared_kwargs)
    return ARED(
        oracle,
        float(kappa),
        int(l_buf_size),
        opts["K_COMP_PTS"],
        opts["QS_VAR"],
        opts["DATA_AUG_VAR"],
        opts["NGHBHOOD_MERGE"],
        opts["SINGLETON_MERGE"],
        opts["SMART_FORGETTING_VAR"],
        opts["VERBOSE_FLAGS"],
    )
