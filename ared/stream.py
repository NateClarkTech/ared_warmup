"""A tiny deterministic stream stored in the JSON config."""

from __future__ import annotations

import numpy as np


def synthetic_stream(seed: int, count: int, dim: int):
    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(int(count), int(dim))).astype(np.float64)
    labels = [f"c{index % 3}" for index in range(int(count))]
    relevance = [(index % 3) == 0 for index in range(int(count))]
    return points, labels, relevance
