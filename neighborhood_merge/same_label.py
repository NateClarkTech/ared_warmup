"""Merge two neighboring clusters that already share a label, and join the second neighbor when it matches."""

from __future__ import annotations


class SameLabelNeighbor:
    name = "same_label"

    def before_query(self, store, hits) -> None:
        if len(hits) < 2:
            return
        first, second = hits[0], hits[1]
        if first.label != second.label or first.cluster_id == second.cluster_id:
            return
        keep = min(first.cluster_id, second.cluster_id)
        drop = max(first.cluster_id, second.cluster_id)
        store.merge(keep, drop)

    def cluster_for_label(self, hits, new_label):
        if len(hits) < 2:
            return None
        if hits[1].label == new_label:
            return hits[1].cluster_id
        return None
