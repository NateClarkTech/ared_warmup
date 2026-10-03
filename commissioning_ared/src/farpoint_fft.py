"""Pool commissioning with global farthest-first queries.

Let L be every labeled pool index and U the unlabeled pool indices. The default
query rule is

    next = argmax_{u in U}  min_{l in L}  ||X[u] - X[l]||_2

The first query is the pool point farthest from the pool centroid (or a uniform
draw when ``first_query="random"``). A newly seen label opens one cluster whose
only labeled point is that query. A repeated label is appended to that class's
labeled points. Remaining unlabeled pool points may be assigned to the nearest
labeled point for prediction only; that assignment never chooses the next query.

``query_rule="legacy_cluster_fp"`` keeps the original per-cluster far-point
loop: the query is the far-point of the cluster with the largest far-point
distance, and a label mismatch partitions that cluster by nearest labeled
point (NQ). It does not run a constrained k-means or density split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
from scipy.spatial.distance import cdist

QUERY_RULE_GLOBAL = "global_fft"
QUERY_RULE_LEGACY = "legacy_cluster_fp"
QUERY_RULES = (QUERY_RULE_GLOBAL, QUERY_RULE_LEGACY)

FIRST_QUERY_CENTROID = "farthest_centroid"
FIRST_QUERY_RANDOM = "random"
FIRST_QUERIES = (FIRST_QUERY_CENTROID, FIRST_QUERY_RANDOM)

RngLike = Union[int, np.integer, np.random.Generator]


@dataclass(eq=False)
class CommissioningState:
    """Everything the streaming detector needs at the pool/watch handoff.

    ``labeled_indices`` are absolute indices into the array that was passed to
    the commissioner. Pass the full stream (and ``pool_end=T``) so those
    indices are absolute stream positions.
    """

    labeled_indices: List[int]
    labels: List[Any]
    relevances: List[bool]
    cluster_dict: Dict[int, Dict[str, Any]]
    known_relevant_labels: Set[Any]
    known_labels: Set[Any]
    prototypes: np.ndarray
    queries_used: int
    classes_discovered: List[Any]
    pool_end: int
    query_rule: str = QUERY_RULE_GLOBAL
    nq_assignment: Optional[np.ndarray] = None

    def summary(self) -> Dict[str, Any]:
        rel_found = [c for c in self.classes_discovered if c in self.known_relevant_labels]
        return {
            "queries_used": int(self.queries_used),
            "classes_discovered": list(self.classes_discovered),
            "relevant_classes_discovered": rel_found,
            "n_classes_discovered": len(self.classes_discovered),
            "n_relevant_discovered": len(rel_found),
            "known_labels": sorted((str(c) for c in self.known_labels)),
            "known_relevant_labels": sorted((str(c) for c in self.known_relevant_labels)),
            "query_rule": self.query_rule,
            "pool_end": int(self.pool_end),
            "labeled_indices": [int(i) for i in self.labeled_indices],
            "labels": [str(y) for y in self.labels],
            "relevances": [bool(r) for r in self.relevances],
        }


def _as_rng(rng: Optional[RngLike]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(f"rng must be a numpy Generator or an integer seed, got {type(rng)!r}")


def min_distances_to_labeled(
    X: np.ndarray,
    unlabeled: np.ndarray,
    labeled: np.ndarray,
    chunk: int = 8192,
) -> np.ndarray:
    """Min Euclidean distance from each unlabeled row to the labeled set.

    Vectorized stand-in for the per-cluster double Python loop in
    ``Cluster.update_far_point``: rows are candidates, columns are labeled
    points, and the column-wise minimum is the distance to the labeled set.
    """
    if len(unlabeled) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(labeled) == 0:
        raise ValueError("labeled set is empty")
    labeled_pts = X[labeled]
    mins = np.empty(len(unlabeled), dtype=np.float64)
    for start in range(0, len(unlabeled), chunk):
        stop = min(start + chunk, len(unlabeled))
        block = cdist(X[unlabeled[start:stop]], labeled_pts, metric="euclidean")
        mins[start:stop] = block.min(axis=1)
    return mins


def farthest_from_centroid(X: np.ndarray, indices: np.ndarray) -> int:
    """Index (into ``X``) of the point in ``indices`` farthest from their centroid."""
    if len(indices) == 0:
        raise ValueError("no candidate indices")
    pts = X[indices]
    centroid = pts.mean(axis=0, keepdims=True)
    dists = cdist(pts, centroid, metric="euclidean").ravel()
    return int(indices[int(np.argmax(dists))])


class _LegacyCluster:
    """One pool cluster for the original per-cluster far-point rule."""

    def __init__(self, label: Optional[str], relevance: bool, l_pts: Sequence[int], o_pts: Sequence[int]):
        self.label = label
        self.relevance = bool(relevance)
        self.l_pts = [int(i) for i in l_pts]
        self.o_pts = [int(i) for i in o_pts]
        self.f_pt = -1
        self.f_pt_dist = 0.0

    def update_far_point(self, X: np.ndarray) -> None:
        """Local max-min far point, vectorized.

        Far point = unlabeled member whose nearest labeled member is as far as
        possible. An empty unlabeled set means the cluster is exhausted.
        """
        if not self.o_pts:
            self.f_pt = -1
            self.f_pt_dist = 0.0
            return
        if not self.l_pts:
            self.f_pt = int(self.o_pts[0])
            self.f_pt_dist = 0.0
            return
        o_idx = np.asarray(self.o_pts, dtype=int)
        l_idx = np.asarray(self.l_pts, dtype=int)
        mins = min_distances_to_labeled(X, o_idx, l_idx)
        local = int(np.argmax(mins))
        self.f_pt = int(o_idx[local])
        self.f_pt_dist = float(mins[local])

    def incorporate_far_point(self, X: np.ndarray) -> None:
        self.o_pts.remove(self.f_pt)
        self.l_pts.append(self.f_pt)
        self.update_far_point(X)


def _nq_split(X: np.ndarray, cluster: _LegacyCluster, f_pt: int, new_label: str, new_relevance: bool):
    """Partition ``cluster`` by nearest labeled seed. Does not choose queries.

    Seeds are the queried far point (new label) followed by the cluster's
    already-labeled points (old label). Each other point follows its nearest
    seed. Points that follow the far point become the new cluster; the rest
    stay with the old label. This is nearest-queried assignment, not a
    k-means bisection of the feature space.
    """
    o_wo = [i for i in cluster.o_pts if i != f_pt]
    seeds = [int(f_pt)] + list(cluster.l_pts)
    if o_wo:
        dists = cdist(X[np.asarray(o_wo, dtype=int)], X[np.asarray(seeds, dtype=int)], metric="euclidean")
        nearest = np.argmin(dists, axis=1)
        fp_o = [o_wo[i] for i in range(len(o_wo)) if int(nearest[i]) == 0]
        old_o = [o_wo[i] for i in range(len(o_wo)) if int(nearest[i]) != 0]
    else:
        fp_o = []
        old_o = []
    fp_cluster = _LegacyCluster(new_label, new_relevance, [int(f_pt)], fp_o)
    old_cluster = _LegacyCluster(cluster.label, cluster.relevance, list(cluster.l_pts), old_o)
    fp_cluster.update_far_point(X)
    old_cluster.update_far_point(X)
    return fp_cluster, old_cluster


class FarpointCommissioner:
    """Spend a label budget on a commissioning pool.

    Parameters
    ----------
    X, y_labels, y_relevance:
        Aligned pool, or the full stream. Only indices in ``[0, pool_end)``
        are eligible. Labels are read from ``y_labels`` (string ``"quit"``
        stops early). Relevance is stored per query and on the cluster.
    budget_B0:
        Maximum number of oracle labels.
    rng:
        Numpy Generator or integer seed. Used for a random first query and
        for the legacy anchor point.
    query_rule:
        ``"global_fft"`` (default) or ``"legacy_cluster_fp"``.
    pool_end:
        Exclusive end of the commissioning pool. Defaults to ``len(X)``.
    first_query:
        ``"farthest_centroid"`` (default) or ``"random"``. Ignored by the
        legacy rule, which keeps the original random-anchor far point.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_labels: Sequence[Any],
        y_relevance: Sequence[Any],
        budget_B0: int,
        rng: Optional[RngLike] = None,
        query_rule: str = QUERY_RULE_GLOBAL,
        pool_end: Optional[int] = None,
        first_query: str = FIRST_QUERY_CENTROID,
    ):
        if query_rule not in QUERY_RULES:
            raise ValueError(f"query_rule must be one of {QUERY_RULES}, got {query_rule!r}")
        if first_query not in FIRST_QUERIES:
            raise ValueError(f"first_query must be one of {FIRST_QUERIES}, got {first_query!r}")
        X_arr = np.array(X, dtype=np.float64, copy=True)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X_arr.shape}")
        n = int(X_arr.shape[0])
        if len(y_labels) != n or len(y_relevance) != n:
            raise ValueError(
                f"y_labels ({len(y_labels)}) and y_relevance ({len(y_relevance)}) "
                f"must match X rows ({n})"
            )
        if pool_end is None:
            pool_end = n
        pool_end = int(pool_end)
        if not 0 <= pool_end <= n:
            raise ValueError(f"pool_end must be in [0, {n}], got {pool_end}")
        budget_B0 = int(budget_B0)
        if budget_B0 < 0:
            raise ValueError(f"budget_B0 must be non-negative, got {budget_B0}")

        self.X = X_arr
        self.y = [str(v) for v in y_labels]
        self.relevance = [bool(v) for v in y_relevance]
        self.budget_B0 = budget_B0
        self.rng = _as_rng(rng)
        self.query_rule = query_rule
        self.pool_end = pool_end
        self.first_query = first_query

        self.labeled_indices: List[int] = []
        self.labels: List[str] = []
        self.relevances: List[bool] = []
        self.classes_discovered: List[str] = []
        self.known_labels: Set[str] = set()
        self.known_relevant_labels: Set[str] = set()
        self.cluster_dict: Dict[int, Dict[str, Any]] = {}
        self._class_to_cluster: Dict[str, int] = {}
        self._next_cid = 0
        self.queries_used = 0
        self._unlabeled = np.ones(pool_end, dtype=bool)
        self._cached_first: Optional[int] = None

        self._leg_clusters: Optional[List[_LegacyCluster]] = None
        self._legacy_have_queried = False

    def _unlabeled_indices(self) -> np.ndarray:
        return np.flatnonzero(self._unlabeled)

    def _read(self, index: int):
        return self.y[index], self.relevance[index]

    def _append_record(self, index: int, label: str, relevance: bool) -> None:
        self.labeled_indices.append(int(index))
        self.labels.append(label)
        self.relevances.append(bool(relevance))
        if 0 <= index < self.pool_end:
            self._unlabeled[index] = False
        self.known_labels.add(label)
        if relevance:
            self.known_relevant_labels.add(label)
        if label not in self.classes_discovered:
            self.classes_discovered.append(label)

    def _initial_index(self) -> int:
        if self._cached_first is not None:
            return self._cached_first
        pool = self._unlabeled_indices()
        if len(pool) == 0:
            raise RuntimeError("commissioning pool is empty")
        if self.first_query == FIRST_QUERY_RANDOM:
            pick = int(pool[int(self.rng.integers(0, len(pool)))])
        else:
            pick = farthest_from_centroid(self.X, pool)
        self._cached_first = int(pick)
        return self._cached_first

    def _global_next_query_index(self) -> int:
        if self.queries_used >= self.budget_B0:
            return -1
        unlabeled = self._unlabeled_indices()
        if len(unlabeled) == 0:
            return -1
        if not self.labeled_indices:
            return self._initial_index()
        labeled = np.asarray(self.labeled_indices, dtype=int)
        mins = min_distances_to_labeled(self.X, unlabeled, labeled)
        return int(unlabeled[int(np.argmax(mins))])

    def _incorporate_global(self, index: int, label: str, relevance: bool) -> None:
        """Open a cluster for a new class, or append to that class's L-pts.

        Does not bisect the pool. Prediction-time nearest-labeled assignment
        is computed only when the state is exported.
        """
        self._append_record(index, label, relevance)
        if label not in self._class_to_cluster:
            cid = self._next_cid
            self._next_cid += 1
            self._class_to_cluster[label] = cid
            self.cluster_dict[cid] = {
                "label": label,
                "relevance": bool(relevance),
                "l_pt_idxs": [int(index)],
            }
        else:
            cid = self._class_to_cluster[label]
            self.cluster_dict[cid]["l_pt_idxs"].append(int(index))
            if relevance:
                self.cluster_dict[cid]["relevance"] = True
        self.queries_used += 1

    def _init_legacy_anchor(self) -> None:
        if self.pool_end <= 0:
            self._leg_clusters = []
            return
        anchor = int(self.rng.integers(0, self.pool_end))
        others = [i for i in range(self.pool_end) if i != anchor]
        cluster = _LegacyCluster(None, False, [anchor], others)
        if not others:
            cluster.f_pt = anchor
            # Positive so the first (only) point is still queried.
            cluster.f_pt_dist = np.inf
        else:
            cluster.update_far_point(self.X)
            if cluster.f_pt < 0:
                cluster.f_pt = anchor
                cluster.f_pt_dist = np.inf
        self._leg_clusters = [cluster]
        self._legacy_have_queried = False

    def _legacy_apply(self, cluster_index: int, query: int, label: str, relevance: bool) -> None:
        assert self._leg_clusters is not None
        cluster = self._leg_clusters[cluster_index]
        if cluster.label is None:
            anchor = cluster.l_pts[0]
            if anchor != query:
                cluster.o_pts.append(anchor)
                cluster.l_pts.remove(anchor)
            cluster.label = label
            cluster.relevance = bool(relevance)
            if query in cluster.o_pts:
                cluster.f_pt = int(query)
                cluster.incorporate_far_point(self.X)
            else:
                # Single-point pool: the anchor itself was the query.
                cluster.l_pts = [int(query)]
                cluster.o_pts = []
                cluster.update_far_point(self.X)
        elif cluster.label == label:
            cluster.incorporate_far_point(self.X)
            if relevance:
                cluster.relevance = True
        else:
            fp_cluster, old_cluster = _nq_split(self.X, cluster, int(query), label, relevance)
            self._leg_clusters.pop(cluster_index)
            self._leg_clusters.append(old_cluster)
            self._leg_clusters.append(fp_cluster)
        self._append_record(query, label, relevance)
        self.queries_used += 1

    def _sync_cluster_dict_from_legacy(self) -> None:
        self.cluster_dict = {}
        self._class_to_cluster = {}
        self._next_cid = 0
        if not self._leg_clusters:
            return
        for cluster in self._leg_clusters:
            if cluster.label is None or not cluster.l_pts:
                continue
            cid = self._next_cid
            self._next_cid += 1
            self.cluster_dict[cid] = {
                "label": cluster.label,
                "relevance": bool(cluster.relevance),
                "l_pt_idxs": [int(i) for i in cluster.l_pts],
            }
            # Legacy can hold several clusters with one label; remember the first.
            self._class_to_cluster.setdefault(cluster.label, cid)

    def _legacy_next_query_index(self) -> int:
        if self.queries_used >= self.budget_B0:
            return -1
        if self._leg_clusters is None:
            self._init_legacy_anchor()
        assert self._leg_clusters is not None
        if not self._leg_clusters:
            return -1
        if not self._legacy_have_queried:
            return int(self._leg_clusters[0].f_pt)
        dists = [c.f_pt_dist for c in self._leg_clusters]
        if max(dists) <= 0:
            return -1
        owner = int(np.argmax(np.asarray(dists, dtype=float)))
        return int(self._leg_clusters[owner].f_pt)

    def _run_legacy(self) -> CommissioningState:
        if self._leg_clusters is None:
            self._init_legacy_anchor()
        assert self._leg_clusters is not None
        while self.queries_used < self.budget_B0:
            if not self._leg_clusters:
                break
            if not self._legacy_have_queried:
                owner, query = 0, int(self._leg_clusters[0].f_pt)
            else:
                dists = [c.f_pt_dist for c in self._leg_clusters]
                if max(dists) <= 0:
                    break
                owner = int(np.argmax(np.asarray(dists, dtype=float)))
                query = int(self._leg_clusters[owner].f_pt)
                if query < 0:
                    break
            label, relevance = self._read(query)
            if label == "quit":
                break
            self._legacy_apply(owner, query, label, relevance)
            self._legacy_have_queried = True
        self._sync_cluster_dict_from_legacy()
        return self.export_state()

    def next_query_index(self) -> int:
        """Index of the next pool query, or -1 when the budget or the pool is exhausted.

        Global mode does not mutate the commissioner except to cache a random
        first draw. Legacy mode may cache the random anchor so a later
        ``run`` uses the same first query.
        """
        if self.query_rule == QUERY_RULE_LEGACY:
            return self._legacy_next_query_index()
        return self._global_next_query_index()

    def _nq_assignment(self) -> Optional[np.ndarray]:
        """Nearest-labeled label for every pool index. Not used to pick queries."""
        if not self.labeled_indices or self.pool_end == 0:
            return None
        labeled = np.asarray(self.labeled_indices, dtype=int)
        label_arr = np.asarray(self.labels, dtype=object)
        assignment = np.empty(self.pool_end, dtype=object)
        chunk = 8192
        for start in range(0, self.pool_end, chunk):
            stop = min(start + chunk, self.pool_end)
            dists = cdist(self.X[start:stop], self.X[labeled], metric="euclidean")
            nearest = np.argmin(dists, axis=1)
            assignment[start:stop] = label_arr[nearest]
        return assignment

    def export_state(self) -> CommissioningState:
        if self.labeled_indices:
            prototypes = self.X[np.asarray(self.labeled_indices, dtype=int)].copy()
        else:
            prototypes = np.zeros((0, self.X.shape[1]), dtype=np.float64)
        cluster_dict = {
            int(cid): {
                "label": info["label"],
                "relevance": bool(info["relevance"]),
                "l_pt_idxs": [int(i) for i in info["l_pt_idxs"]],
            }
            for cid, info in self.cluster_dict.items()
        }
        return CommissioningState(
            labeled_indices=[int(i) for i in self.labeled_indices],
            labels=list(self.labels),
            relevances=[bool(r) for r in self.relevances],
            cluster_dict=cluster_dict,
            known_relevant_labels=set(self.known_relevant_labels),
            known_labels=set(self.known_labels),
            prototypes=prototypes,
            queries_used=int(self.queries_used),
            classes_discovered=list(self.classes_discovered),
            pool_end=int(self.pool_end),
            query_rule=self.query_rule,
            nq_assignment=self._nq_assignment(),
        )

    def run(self) -> CommissioningState:
        """Query until the budget is spent, the pool is exhausted, or the label is ``quit``."""
        if self.query_rule == QUERY_RULE_LEGACY:
            return self._run_legacy()
        while self.queries_used < self.budget_B0:
            query = self._global_next_query_index()
            if query < 0:
                break
            label, relevance = self._read(query)
            if label == "quit":
                break
            self._incorporate_global(query, label, relevance)
        return self.export_state()


def commission_from_indices(
    X: np.ndarray,
    y_labels: Sequence[Any],
    y_relevance: Sequence[Any],
    indices: Sequence[int],
    pool_end: Optional[int] = None,
    query_rule: str = "indexed",
) -> CommissioningState:
    """Build a commissioning state from an explicit query list.

    One cluster per class, same incorporation rule as global farthest-first.
    Used by the random and oracle pool protocols. Indices outside the pool or
    repeated indices are skipped. ``"quit"`` stops the list.
    """
    commissioner = FarpointCommissioner(
        X,
        y_labels,
        y_relevance,
        budget_B0=len(indices),
        rng=0,
        query_rule=QUERY_RULE_GLOBAL,
        pool_end=pool_end,
    )
    # The budget above only caps ``run``; incorporation below is explicit.
    seen = set()
    for raw in indices:
        index = int(raw)
        if index in seen:
            continue
        if not 0 <= index < commissioner.pool_end:
            continue
        seen.add(index)
        label, relevance = commissioner._read(index)
        if label == "quit":
            break
        commissioner._incorporate_global(index, label, relevance)
    state = commissioner.export_state()
    state.query_rule = query_rule
    return state
