"""The A/RED algorithm. Variant math lives in the strategy folders this class calls."""

from __future__ import annotations

from comparison_distance import build as build_comparison_distance
from neighborhood_merge import build as build_neighborhood_merge
from singleton_merge import build as build_singleton_merge
from smart_forgetting import build as build_forgetting

import numpy as np


class Memory:
    def __init__(self, vector, label: str, relevant: bool, cluster_id: int):
        self.vector = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
        self.label = str(label)
        self.relevant = bool(relevant)
        self.cluster_id = int(cluster_id)


class Hit:
    def __init__(self, memory: Memory, distance: float):
        self.memory = memory
        self.distance = float(distance)

    @property
    def label(self) -> str:
        return self.memory.label

    @property
    def relevant(self) -> bool:
        return self.memory.relevant

    @property
    def cluster_id(self) -> int:
        return self.memory.cluster_id


class Cluster:
    def __init__(self, cluster_id: int, label: str, relevant: bool):
        self.id = int(cluster_id)
        self.label = str(label)
        self.relevant = bool(relevant)
        self.members: list[Memory] = []
        self.comparison_distance = 0.0

    def vectors(self):
        if not self.members:
            return np.zeros((0, 1), dtype=np.float64)
        return np.vstack([member.vector for member in self.members])


class ClusterStore:
    def __init__(self, comparison):
        self.comparison = comparison
        self._clusters: dict[int, Cluster] = {}
        self._by_label: dict[str, Cluster] = {}
        self._next_id = 0

    def __contains__(self, cluster_id: int) -> bool:
        return cluster_id in self._clusters

    def all(self) -> list[Cluster]:
        return list(self._clusters.values())

    def get(self, cluster_id: int) -> Cluster:
        return self._clusters[cluster_id]

    def open(self, label: str, relevant: bool) -> Cluster:
        cluster = Cluster(self._next_id, label, relevant)
        self._next_id += 1
        self._clusters[cluster.id] = cluster
        self._by_label.setdefault(cluster.label, cluster)
        return cluster

    def cluster_for_label(self, label: str, relevant: bool) -> Cluster:
        cluster = self._by_label.get(label)
        if cluster is None or cluster.id not in self._clusters:
            return self.open(label, relevant)
        if relevant:
            cluster.relevant = True
        return cluster

    def add_member(self, cluster: Cluster, memory: Memory) -> None:
        memory.cluster_id = cluster.id
        cluster.members.append(memory)
        if memory.relevant:
            cluster.relevant = True
        self.remeasure(cluster)

    def detach(self, memory: Memory) -> None:
        cluster = self._clusters.get(memory.cluster_id)
        if cluster is None:
            return
        cluster.members = [member for member in cluster.members if member is not memory]
        if cluster.members:
            self.remeasure(cluster)
            return
        del self._clusters[cluster.id]
        if self._by_label.get(cluster.label) is cluster:
            replacement = next(
                (other for other in self._clusters.values() if other.label == cluster.label),
                None,
            )
            if replacement is None:
                del self._by_label[cluster.label]
            else:
                self._by_label[cluster.label] = replacement

    def merge(self, keep_id: int, drop_id: int) -> None:
        if keep_id == drop_id:
            return
        keep = self._clusters[keep_id]
        drop = self._clusters.pop(drop_id)
        for memory in drop.members:
            memory.cluster_id = keep.id
            keep.members.append(memory)
        if drop.relevant:
            keep.relevant = True
        if self._by_label.get(drop.label) is drop:
            self._by_label[drop.label] = keep
        self.remeasure(keep)

    def remeasure(self, cluster: Cluster) -> None:
        cluster.comparison_distance = float(self.comparison.measure(cluster.vectors()))


class Buffer:
    def __init__(self, capacity: int, forgetting):
        if capacity < 1:
            raise ValueError(f"buffer capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.forgetting = forgetting
        self.items: list[Memory] = []

    def add(self, memory: Memory) -> Memory | None:
        dropped = None
        if len(self.items) >= self.capacity:
            index = self.forgetting.index_to_drop(self.items)
            dropped = self.items.pop(index)
        self.items.append(memory)
        return dropped

    def nearest(self, vector, k: int) -> list[Hit]:
        if not self.items or k < 1:
            return []
        target = np.asarray(vector, dtype=np.float64).reshape(-1)
        distances = [float(np.linalg.norm(item.vector - target)) for item in self.items]
        order = sorted(range(len(self.items)), key=lambda index: (distances[index], index))
        return [Hit(self.items[index], distances[index]) for index in order[:k]]


def should_query(distance: float, kappa: float, comparison_distance: float, nearby_relevant: bool) -> bool:
    """Query when the point is outside the cluster's comparison distance, or a nearby point is relevant."""
    return bool(nearby_relevant) or float(distance) * float(kappa) > float(comparison_distance)


class ARED:
    def __init__(self, points, labels, relevance, kappa, buffer_size, k_neighbors, comparison, forgetting, neighborhood, singleton):
        self.points = np.asarray(points, dtype=np.float64)
        self.labels = [str(value) for value in labels]
        self.relevance = [bool(value) for value in relevance]
        self.kappa = float(kappa)
        self.k_neighbors = int(k_neighbors)
        self.comparison = comparison
        self.neighborhood = neighborhood
        self.singleton = singleton
        self.clusters = ClusterStore(comparison)
        self.buffer = Buffer(buffer_size, forgetting)

    @classmethod
    def from_config(cls, config, points, labels, relevance) -> "ARED":
        return cls(
            points,
            labels,
            relevance,
            kappa=config.kappa,
            buffer_size=config.buffer_size,
            k_neighbors=config.k_neighbors,
            comparison=build_comparison_distance(config.comparison_distance),
            forgetting=build_forgetting(config.smart_forgetting),
            neighborhood=build_neighborhood_merge(config.neighborhood_merge),
            singleton=build_singleton_merge(config.singleton_merge),
        )

    def seed(self, indices) -> None:
        for index in indices:
            self.remember(self.points[index], self.labels[index], self.relevance[index])
        self.singleton.apply(self.clusters)

    def remember(self, vector, label, relevant) -> None:
        """Store one labeled warm-up point, forgetting through the configured policy."""
        cluster = self.clusters.cluster_for_label(str(label), bool(relevant))
        self._store_in(cluster.id, vector, str(label), bool(relevant))

    def stream_point(self, index: int) -> bool:
        """Decide whether stream index ``index`` is queried. True means it was labeled."""
        vector = self.points[index]
        if not self.buffer.items:
            self.remember(vector, self.labels[index], self.relevance[index])
            self.singleton.apply(self.clusters)
            return True

        hits = self.buffer.nearest(vector, self.k_neighbors)
        self.neighborhood.before_query(self.clusters, hits)
        nearest = hits[0]
        comparison = self.clusters.get(nearest.cluster_id)
        nearby_relevant = any(hit.relevant for hit in hits)
        if not should_query(nearest.distance, self.kappa, comparison.comparison_distance, nearby_relevant):
            return False

        label = self.labels[index]
        relevant = self.relevance[index]
        if label == comparison.label:
            target = comparison.id
        else:
            target = self.neighborhood.cluster_for_label(hits, label)
        if target is None or target not in self.clusters:
            target = self.clusters.open(label, relevant).id
        self._store_in(target, vector, label, relevant)
        self.singleton.apply(self.clusters)
        return True

    def buffered_labels(self) -> list[str]:
        return [item.label for item in self.buffer.items]

    def _store_in(self, cluster_id: int, vector, label: str, relevant: bool) -> None:
        cluster = self.clusters.get(cluster_id)
        memory = Memory(vector, label, relevant, cluster.id)
        dropped = self.buffer.add(memory)
        self.clusters.add_member(cluster, memory)
        if dropped is not None:
            self.clusters.detach(dropped)
