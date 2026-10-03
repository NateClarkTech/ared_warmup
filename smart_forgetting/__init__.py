"""Pick which buffered point is forgotten when the window is full."""

from __future__ import annotations

from smart_forgetting.keep_relevant import KeepRelevant
from smart_forgetting.plain import PlainForgetting

STRATEGIES = {
    PlainForgetting.name: PlainForgetting,
    KeepRelevant.name: KeepRelevant,
}


def build(name: str):
    if name not in STRATEGIES:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown smart forgetting {name!r}. Expected one of: {known}")
    return STRATEGIES[name]()
