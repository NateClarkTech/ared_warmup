"""Fold a one-point cluster into the nearest cluster that has the same label."""

from __future__ import annotations

import numpy as np


class MergeSingletons:
    name = "merge_singletons"

    def apply(self, store) -> None:
        merged = True
        while merged:
            merged = False
            for cluster in store.all():
                if cluster.id not in store or len(cluster.members) != 1:
                    continue
                target = _nearest_same_label(store, cluster)
                if target is None:
                    continue
                store.merge(target.id, cluster.id)
                merged = True
                break


def _nearest_same_label(store, cluster):
    center = cluster.vectors().mean(axis=0)
    best = None
    best_distance = None
    for other in store.all():
        if other.id == cluster.id or other.label != cluster.label or not other.members:
            continue
        distance = float(np.linalg.norm(other.vectors().mean(axis=0) - center))
        if best is None or distance < best_distance:
            best = other
            best_distance = distance
    return best
