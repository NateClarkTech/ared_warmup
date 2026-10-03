"""Leave neighboring clusters as they are."""

from __future__ import annotations


class Disabled:
    name = "disabled"

    def before_query(self, store, hits) -> None:
        return None

    def cluster_for_label(self, hits, new_label):
        return None
