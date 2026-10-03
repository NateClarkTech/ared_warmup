"""Farthest-first queries on one prefix.

The first query is the prefix point farthest from the prefix centroid.
Each later query is the unlabeled prefix point whose nearest already-queried
point is as far away as possible.
"""

from __future__ import annotations

import numpy as np


def far_point_indices(prefix) -> list[int]:
    points = np.asarray(prefix, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"prefix must be a 2-d array, got shape {points.shape}")
    count = points.shape[0]
    if count == 0:
        return []

    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    chosen = [first]
    unlabeled = np.ones(count, dtype=bool)
    unlabeled[first] = False

    while unlabeled.any():
        candidates = np.flatnonzero(unlabeled)
        chosen_pts = points[np.asarray(chosen, dtype=int)]
        gaps = points[candidates, None, :] - chosen_pts[None, :, :]
        nearest = np.linalg.norm(gaps, axis=2).min(axis=1)
        pick = int(candidates[int(np.argmax(nearest))])
        chosen.append(pick)
        unlabeled[pick] = False
    return chosen
