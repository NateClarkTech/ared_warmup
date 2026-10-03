"""Comparison distance is the average distance from each point to its nearest neighbor."""

from __future__ import annotations

import numpy as np


class AverageNearestNeighbor:
    name = "average_nearest_neighbor"

    def measure(self, points) -> float:
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for i in range(len(pts)):
            nearest = None
            for j in range(len(pts)):
                if i == j:
                    continue
                gap = float(np.linalg.norm(pts[i] - pts[j]))
                if nearest is None or gap < nearest:
                    nearest = gap
            total += 0.0 if nearest is None else nearest
        return total / len(pts)
