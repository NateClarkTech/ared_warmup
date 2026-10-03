"""Pick how a cluster turns its points into one comparison distance."""

from __future__ import annotations

from comparison_distance.average_nearest_neighbor import AverageNearestNeighbor
from comparison_distance.diameter import Diameter

STRATEGIES = {
    Diameter.name: Diameter,
    AverageNearestNeighbor.name: AverageNearestNeighbor,
}


def build(name: str):
    if name not in STRATEGIES:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown comparison distance {name!r}. Expected one of: {known}")
    return STRATEGIES[name]()
