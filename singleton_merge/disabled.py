"""Leave a one-point cluster alone."""

from __future__ import annotations


class Disabled:
    name = "disabled"

    def apply(self, store) -> None:
        return None
