"""Cold start, random, farthest-first, and oracle commissioning, then A/RED.

All four methods are scored on the watch phase ``[T, N)`` at the same kappa.
Cold start has no pool archive: the detector runs from index 0, and the split
at T is only a metric boundary. The other three spend ``B0`` pool labels on
``X[:T]`` and hand the resulting clusters to the detector, which then calls
``process_point`` from T onward.

Farpoint-on-the-full-pool is not a fifth method. Each record stores how many
relevant classes are present in the pool (the commissioning discovery ceiling).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.datasets import DatasetBundle, load_dataset
from src.farpoint_fft import (
    QUERY_RULE_GLOBAL,
    FarpointCommissioner,
    commission_from_indices,
)
from src.handoff import empty_ared, make_oracle
from src.handoff import ARED  # classmethod from_commissioning_state is attached on import
from src.metrics import (
    QueryLog,
    labels_per_relevant,
    query_precision,
    relevant_recall_and_counts,
    snapshot_ared,
    watch_delta,
)
from src.streams import (
    apply_permutation,
    assert_late_arrival,
    assert_stationary,
    late_arrival_permutation,
    stationary_permutation,
)

METHODS = ("cold_ared", "random_then_ared", "farpoint_then_ared", "oracle_then_ared")


def default_n_rel(data: str) -> int:
    if data.strip().upper() == "PARKING_LOT_DINO":
        return 6
    return 4


def default_kappa(data: str) -> float:
    if data.strip().upper() == "PARKING_LOT_DINO":
        return 0.5
    if data.strip().upper() == "NICE":
        return 1.0
    return 0.5


def pool_length(n: int, t_frac: float) -> int:
    if n < 2:
        raise ValueError(f"stream must contain at least 2 points, got {n}")
    if not 0.0 < float(t_frac) < 1.0:
        raise ValueError(f"t_frac must be in (0, 1), got {t_frac}")
    T = int(round(float(t_frac) * n))
    return int(min(max(T, 1), n - 1))


def _ared_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    smart = cfg.get("smart_forgetting", (3, 0.01))
    return {
        "K_COMP_PTS": int(cfg.get("k_comp_pts", 2)),
        "QS_VAR": int(cfg.get("qs_var", 1)),
        "DATA_AUG_VAR": tuple(cfg.get("data_aug_var", (0, (0,)))),
        "NGHBHOOD_MERGE": bool(cfg.get("neighborhood_merge", False)),
        "SINGLETON_MERGE": bool(cfg.get("singleton_merge", False)),
        "SMART_FORGETTING_VAR": (int(smart[0]), float(smart[1])),
        "VERBOSE_FLAGS": list(cfg.get("verbose_flags", [])),
    }


def build_stream(labels: np.ndarray, rel_classes: Sequence[str], T: int, stream: str, rng: np.random.Generator):
    kind = stream.strip().lower()
    if kind == "stationary":
        perm = stationary_permutation(labels, T, rng)
        ordered = labels[perm]
        assert_stationary(ordered, T)
        return perm, None
    if kind == "late_arrival":
        perm, held = late_arrival_permutation(labels, rel_classes, T, rng)
        ordered = labels[perm]
        assert_late_arrival(ordered, T, held)
        return perm, held
    raise ValueError(f"stream must be 'stationary' or 'late_arrival', got {stream!r}")


def _random_indices(T: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    take = min(int(budget), int(T))
    if take <= 0:
        return np.zeros((0,), dtype=int)
    return np.asarray(rng.choice(int(T), size=take, replace=False), dtype=int)


def _oracle_indices(labels: Sequence[str], T: int, budget: int, rng: np.random.Generator) -> list:
    """One example of each pool class, rarest first, then uniform leftovers."""
    pool = [str(v) for v in list(labels)[:T]]
    from collections import Counter

    counts = Counter(pool)
    order = sorted(counts, key=lambda label: (counts[label], label))
    chosen = []
    used = set()
    for label in order:
        if len(chosen) >= budget:
            break
        candidates = [i for i, value in enumerate(pool) if value == label]
        pick = int(rng.choice(np.asarray(candidates, dtype=int)))
        chosen.append(pick)
        used.add(pick)
    if len(chosen) < budget:
        rest = [i for i in range(T) if i not in used]
        extra = min(budget - len(chosen), len(rest))
        if extra:
            picks = rng.choice(np.asarray(rest, dtype=int), size=extra, replace=False)
            chosen.extend(int(i) for i in np.atleast_1d(picks))
    return chosen


def _commission(method: str, X, labels, relevance, T: int, budget: int, rng, query_rule: str, first_query: str):
    if method == "cold_ared":
        return None
    if method == "farpoint_then_ared":
        commissioner = FarpointCommissioner(
            X,
            labels,
            relevance,
            budget_B0=budget,
            rng=rng,
            query_rule=query_rule,
            pool_end=T,
            first_query=first_query,
        )
        return commissioner.run()
    if method == "random_then_ared":
        indices = _random_indices(T, budget, rng)
        return commission_from_indices(X, labels, relevance, indices, pool_end=T, query_rule="random")
    if method == "oracle_then_ared":
        indices = _oracle_indices(labels, T, budget, rng)
        return commission_from_indices(X, labels, relevance, indices, pool_end=T, query_rule="oracle")
    raise ValueError(f"Unknown method {method!r}. Expected one of {METHODS}")


def _feed(ared, X, index: int, log: QueryLog, labels, relevance) -> None:
    before = int(ared.num_queries)
    n_buf = int(ared.l_buf.data_circular_buffer.count)
    if n_buf == 0 and int(ared.num_pts_streamed) == index:
        ared.process_first_point(np.asarray(X[index], dtype=np.float64))
    else:
        ared.process_point(np.asarray(X[index], dtype=np.float64))
    if int(ared.num_pts_streamed) != index + 1:
        raise RuntimeError(
            f"stream counter drifted at index {index}: num_pts_streamed={ared.num_pts_streamed}"
        )
    if int(ared.num_queries) > before:
        log.add(index, labels[index], relevance[index])


def _run_range(ared, X, labels, relevance, start: int, end: int, show_progress: bool, desc: str) -> QueryLog:
    """Feed ``X[start:end]`` in order. ``start`` must equal ``ared.num_pts_streamed``."""
    log = QueryLog()
    iterator = range(int(start), int(end))
    if show_progress and (end - start) >= 400:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=desc, leave=False)
    for index in iterator:
        _feed(ared, X, int(index), log, labels, relevance)
    return log


def _json_latency(latency: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    return {str(k): (None if v is None else int(v)) for k, v in latency.items()}


def run_configured(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run one method on one stream construction. Returns a JSON-ready record."""
    data_name = str(cfg.get("data", "NICE"))
    method = str(cfg["method"])
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Expected one of {METHODS}")
    seed = int(cfg.get("seed", 0))
    t_frac = float(cfg.get("t_frac", 0.2))
    budget = int(cfg.get("B0", cfg.get("budget_B0", 20)))
    n_rel = cfg.get("n_rel_classes")
    if n_rel is None:
        n_rel = default_n_rel(data_name)
    n_rel = int(n_rel)
    kappa = cfg.get("kappa")
    if kappa is None:
        kappa = default_kappa(data_name)
    kappa = float(kappa)
    l_buf = int(cfg.get("l_buf_size", 1000))
    stream = str(cfg.get("stream", "stationary"))
    query_rule = str(cfg.get("query_rule", QUERY_RULE_GLOBAL))
    first_query = str(cfg.get("first_query", "farthest_centroid"))
    nice_base = int(cfg.get("nice_base", 800))
    nice_min = int(cfg.get("nice_min_count", 20))
    verbose = bool(cfg.get("verbose", True))
    show_progress = bool(cfg.get("progress", verbose))

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    bundle: DatasetBundle = cfg.get("bundle") or load_dataset(
        data_name,
        n_rel_classes=n_rel,
        seed=seed,
        nice_base=nice_base,
        nice_min_count=nice_min,
        verbose=verbose,
    )
    # Relevance classes come from the bundle so a preloaded bundle and a fresh
    # load follow the same rarest-n rule.
    rel_classes = [str(c) for c in bundle.rel_classes]
    T = pool_length(len(bundle.X), t_frac)
    stream_rng = np.random.default_rng(seed + 17)
    perm, held = build_stream(bundle.labels, rel_classes, T, stream, stream_rng)
    X, labels, relevance = apply_permutation(bundle.X, bundle.labels, bundle.relevance, perm)
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray([str(v) for v in labels.tolist()], dtype=object)
    relevance = np.asarray(relevance, dtype=bool)

    if budget < 0:
        raise ValueError(f"B0 must be non-negative, got {budget}")
    budget_eff = min(budget, T)

    pool_labels = set(labels[:T].tolist())
    rel_in_pool = [c for c in rel_classes if c in pool_labels]
    post_rel = relevance[T:]
    post_labels = labels[T:]
    if post_rel.any():
        discoverable = float(np.mean([lab in pool_labels for lab, flag in zip(post_labels, post_rel) if flag]))
    else:
        discoverable = 0.0

    started = time.perf_counter()
    comm_rng = np.random.default_rng(seed + 101)
    state = _commission(
        method, X, labels, relevance, T, budget_eff, comm_rng, query_rule, first_query
    )
    oracle = make_oracle(X, labels, relevance)
    options = _ared_options(cfg)
    comm_log = QueryLog()
    if state is not None:
        comm_log.extend_pairs(state.labeled_indices, state.labels, state.relevances)

    if method == "cold_ared":
        ared = empty_ared(oracle, kappa, l_buf, **options)
        pre_log = _run_range(ared, X, labels, relevance, 0, T, show_progress, f"{method} pool")
        at_T = snapshot_ared(ared)
        watch_log = _run_range(ared, X, labels, relevance, T, len(X), show_progress, f"{method} watch")
        # pre_log is the cold "commissioning" queries; keep it for discovery.
        comm_log = pre_log
        cold_classes = []
        seen = set()
        for _idx, label, _rel in pre_log.events:
            if label not in seen:
                seen.add(label)
                cold_classes.append(label)
        commissioning = {
            "queries_used": len(pre_log.events),
            "classes_discovered": cold_classes,
            "relevant_classes_discovered": [c for c in cold_classes if c in set(rel_classes)],
            "n_classes_discovered": len(cold_classes),
            "n_relevant_discovered": len([c for c in cold_classes if c in set(rel_classes)]),
            "query_rule": None,
            "pool_end": T,
            "labeled_indices": [int(i) for i, _y, _r in pre_log.events],
            "labels": [y for _i, y, _r in pre_log.events],
            "relevances": [bool(r) for _i, _y, r in pre_log.events],
        }
    else:
        ared = ARED.from_commissioning_state(
            state, oracle, kappa, l_buf, t_index=T, **options
        )
        at_T = snapshot_ared(ared)
        watch_log = _run_range(ared, X, labels, relevance, T, len(X), show_progress, f"{method} watch")
        commissioning = state.summary()

    at_end = snapshot_ared(ared)
    delta = watch_delta(at_T, at_end)
    recall, n_queried, n_streamed = relevant_recall_and_counts(delta["conf_matrix"], rel_classes, ared)
    qp = query_precision(delta["num_correct_queries"], delta["num_queries"])
    missed = max(0.0, n_streamed - n_queried)
    missed_frac = float(missed / n_streamed) if n_streamed else 0.0
    watch_n = max(len(X) - T, 1)

    combined = QueryLog()
    combined.events.extend(comm_log.events)
    combined.events.extend(watch_log.events)
    latency = combined.discovery_latency(rel_classes)
    known_irrel_before_watch = set()
    if state is not None:
        for info in state.cluster_dict.values():
            if not info["relevance"]:
                known_irrel_before_watch.add(str(info["label"]))
    elif method == "cold_ared":
        # Classes observed before T with a negative relevance bit.
        seen_rel = {}
        for _i, label, rel in comm_log.events:
            seen_rel.setdefault(label, bool(rel))
            if rel:
                seen_rel[label] = True
        known_irrel_before_watch = {label for label, rel in seen_rel.items() if not rel}

    total_labels = len(combined.events)
    rel_found = combined.relevant_found()
    elapsed = time.perf_counter() - started
    record = {
        "method": method,
        "data": bundle.name,
        "stream": stream,
        "seed": seed,
        "t_frac": t_frac,
        "T": int(T),
        "N": int(len(X)),
        "B0": int(budget),
        "B0_effective": int(budget_eff if method != "cold_ared" else 0),
        "kappa": kappa,
        "l_buf_size": l_buf,
        "n_rel_classes": n_rel,
        "query_rule": query_rule if method == "farpoint_then_ared" else None,
        "qs_var": options["QS_VAR"],
        "k_comp_pts": options["K_COMP_PTS"],
        "smart_forgetting": list(options["SMART_FORGETTING_VAR"]),
        "rel_classes": rel_classes,
        "held_out_class": held,
        "n_rel_classes_in_pool": len(rel_in_pool),
        "rel_classes_in_pool": rel_in_pool,
        "post_T_rel_mass_from_pool_classes": discoverable,
        "seconds": round(elapsed, 4),
        "commissioning": commissioning,
        "watch": {
            "query_precision": qp,
            "relevant_recall": recall,
            "num_queries": int(delta["num_queries"]),
            "num_correct_queries": int(delta["num_correct_queries"]),
            "query_rate": float(delta["num_queries"]) / float(watch_n),
            "relevant_queried": n_queried,
            "relevant_streamed": n_streamed,
            "missed_relevant_mass": missed,
            "missed_relevant_fraction": missed_frac,
            "discovery_latency": _json_latency(latency),
            "queries_on_known_irrelevant": watch_log.queries_on_known_irrelevant(known_irrel_before_watch),
            "cumulative_relevant_seen": int(delta["cumulative_relevant_seen"]),
        },
        "whole": {
            "total_labels": int(total_labels),
            "relevant_examples_found": int(rel_found),
            "labels_per_relevant_found": labels_per_relevant(total_labels, rel_found),
            "post_T_query_rate": float(delta["num_queries"]) / float(watch_n),
            "commissioning_queries_on_known_irrelevant": comm_log.queries_on_known_irrelevant(),
        },
    }
    if verbose:
        _print_record(record)
    return record


def _print_record(record: Dict[str, Any]) -> None:
    comm = record["commissioning"]
    watch = record["watch"]
    whole = record["whole"]
    print()
    print(
        f"{record['method']}  stream={record['stream']}  seed={record['seed']}  "
        f"T={record['T']}/{record['N']}  B0={record['B0']}  kappa={record['kappa']}  "
        f"l_buf={record['l_buf_size']}"
        + (f"  held_out={record['held_out_class']}" if record.get("held_out_class") else "")
    )
    print(
        f"  commissioning queries={comm['queries_used']}  "
        f"classes={comm['classes_discovered']}  "
        f"relevant={comm['relevant_classes_discovered']}"
    )
    print(
        f"  watch Query_Precision={watch['query_precision']:.4f}  "
        f"Relevant_Recall={watch['relevant_recall']:.4f}  "
        f"watch_queries={watch['num_queries']}  "
        f"post_T_query_rate={watch['query_rate']:.4f}"
    )
    print(
        f"  discovery_latency={watch['discovery_latency']}  "
        f"queries_on_known_irrelevant={watch['queries_on_known_irrelevant']}  "
        f"missed_relevant_mass={watch['missed_relevant_mass']:.0f}"
    )
    lpr = whole["labels_per_relevant_found"]
    lpr_txt = "n/a" if lpr is None else f"{lpr:.3f}"
    print(
        f"  total_labels={whole['total_labels']}  "
        f"relevant_found={whole['relevant_examples_found']}  "
        f"labels_per_relevant_found={lpr_txt}  "
        f"seconds={record['seconds']:.2f}"
    )
