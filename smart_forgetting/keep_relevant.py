"""Drop the oldest point that is not relevant. A relevant label stays."""

from __future__ import annotations


class KeepRelevant:
    name = "keep_relevant"

    def index_to_drop(self, items) -> int:
        if not items:
            raise ValueError("the buffer is empty")
        for index, item in enumerate(items):
            if not item.relevant:
                return index
        # The buffer is entirely relevant, so the oldest point has to leave.
        return 0
