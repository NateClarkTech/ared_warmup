"""Comparison distance is the farthest pair in the cluster."""

from __future__ import annotations

import numpy as np


class Diameter:
    name = "diameter"

    def measure(self, points) -> float:
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            return 0.0
        widest = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                gap = float(np.linalg.norm(pts[i] - pts[j]))
                if gap > widest:
                    widest = gap
        return widest
