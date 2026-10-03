"""Pick what happens when a point's nearest buffered neighbors disagree about the cluster."""

from __future__ import annotations

from neighborhood_merge.disabled import Disabled
from neighborhood_merge.same_label import SameLabelNeighbor

STRATEGIES = {
    Disabled.name: Disabled,
    SameLabelNeighbor.name: SameLabelNeighbor,
}


def build(name: str):
    if name not in STRATEGIES:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown neighborhood merge {name!r}. Expected one of: {known}")
    return STRATEGIES[name]()
