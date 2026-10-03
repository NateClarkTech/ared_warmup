"""Watch-phase metrics.

Query precision and relevant recall use the vendored definitions:

* query precision = ``num_correct_queries / num_queries``
  (a query counts as correct inside the detector when the queried point is
  relevant; see ``ARED.process_point``)
* relevant recall = queried relevant points / streamed relevant points,
  read off the confusion matrix by ``calculate_single_rel_recall``
  (diagonal mass on relevant classes over the full relevant row sum)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.vendor_bootstrap import import_ared_symbols

_, _, calculate_single_rel_recall = import_ared_symbols()


def query_precision(num_correct: float, num_queries: float) -> float:
    if num_queries <= 0:
        return 0.0
    return float(num_correct) / float(num_queries)


def relevant_recall_and_counts(confusion: np.ndarray, rel_classes: Sequence[str], ared) -> Tuple[float, float, float]:
    """Return ``(recall, n_queried, n_streamed)`` for the relevant classes.

    Recall is the vendored scalar. The two counts use the same cells that
    function reads: the diagonal (queried) and the row sum (streamed).
    """
    recall, n_streamed = calculate_single_rel_recall(confusion, list(rel_classes), ared)
    cm = np.asarray(confusion)
    n_queried = 0.0
    for label in rel_classes:
        index = ared.oracle.int_str_label_bidict[str(label)]
        n_queried += float(cm[index, index])
    return float(recall), float(n_queried), float(n_streamed)


def snapshot_ared(ared) -> Dict[str, Any]:
    return {
        "num_queries": int(ared.num_queries),
        "num_correct_queries": int(ared.num_correct_queries),
        "conf_matrix": np.array(ared.conf_matrix, copy=True),
        "cumulative_relevant_seen": int(ared.cumulative_relevant_seen),
        "num_pts_streamed": int(ared.num_pts_streamed),
    }


def watch_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_queries": int(after["num_queries"] - before["num_queries"]),
        "num_correct_queries": int(after["num_correct_queries"] - before["num_correct_queries"]),
        "conf_matrix": after["conf_matrix"] - before["conf_matrix"],
        "cumulative_relevant_seen": int(
            after["cumulative_relevant_seen"] - before["cumulative_relevant_seen"]
        ),
        "num_pts_streamed": int(after["num_pts_streamed"] - before["num_pts_streamed"]),
    }


class QueryLog:
    """Oracle queries in time order. Each entry is ``(stream_index, label, relevant)``."""

    def __init__(self):
        self.events: List[Tuple[int, str, bool]] = []

    def add(self, index: int, label: Any, relevant: Any) -> None:
        self.events.append((int(index), str(label), bool(relevant)))

    def extend_pairs(self, indices: Sequence[int], labels: Sequence[Any], relevances: Sequence[Any]) -> None:
        for index, label, rel in zip(indices, labels, relevances):
            self.add(index, label, rel)

    def discovery_latency(self, rel_classes: Sequence[str]) -> Dict[str, Optional[int]]:
        """Absolute stream index of the first query of each relevant class."""
        latency: Dict[str, Optional[int]] = {str(label): None for label in rel_classes}
        for index, label, _rel in self.events:
            if label in latency and latency[label] is None:
                latency[label] = int(index)
        return latency

    def queries_on_known_irrelevant(self, already_irrelevant: Optional[Iterable[str]] = None) -> int:
        """Queries whose label was already known to be irrelevant before the query."""
        known = {str(label) for label in (already_irrelevant or ())}
        count = 0
        for _index, label, relevant in self.events:
            if label in known:
                count += 1
            if not relevant:
                known.add(label)
        return int(count)

    def relevant_found(self) -> int:
        return int(sum(1 for _i, _y, rel in self.events if rel))


def labels_per_relevant(total_labels: int, relevant_found: int) -> Optional[float]:
    if relevant_found <= 0:
        return None
    return float(total_labels) / float(relevant_found)
