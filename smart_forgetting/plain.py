"""Drop the oldest buffered point."""

from __future__ import annotations


class PlainForgetting:
    name = "plain"

    def index_to_drop(self, items) -> int:
        if not items:
            raise ValueError("the buffer is empty")
        return 0
