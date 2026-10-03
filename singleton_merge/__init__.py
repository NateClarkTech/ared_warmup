"""Pick whether a one-point cluster is absorbed by a same-label neighbor."""

from __future__ import annotations

from singleton_merge.disabled import Disabled
from singleton_merge.merge_singletons import MergeSingletons

STRATEGIES = {
    Disabled.name: Disabled,
    MergeSingletons.name: MergeSingletons,
}


def build(name: str):
    if name not in STRATEGIES:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown singleton merge {name!r}. Expected one of: {known}")
    return STRATEGIES[name]()
