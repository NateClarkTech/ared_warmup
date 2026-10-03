"""Index permutations of a fixed pool.

Stationary streams place every class that has at least two examples on both
sides of T. Late arrival holds every example of one relevant class (the
rarest that fits) until after T.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, Tuple

import numpy as np


def _as_labels(labels: Sequence) -> np.ndarray:
    return np.asarray([str(v) for v in labels], dtype=object)


def _swap(perm: np.ndarray, lab: np.ndarray, i: int, j: int) -> None:
    perm[i], perm[j] = perm[j], perm[i]
    lab[i], lab[j] = lab[j], lab[i]


def _donor(lab: np.ndarray, start: int, end: int, blocked: str) -> Optional[int]:
    segment = lab[start:end].tolist()
    counts = Counter(segment)
    for offset, label in enumerate(segment):
        if label == blocked:
            continue
        if counts[label] >= 2:
            return start + offset
    return None


def repair_stationary(perm: np.ndarray, labels_in_perm_order: np.ndarray, T: int) -> np.ndarray:
    """Swap until every class with >= 2 points appears in ``[0, T)`` and ``[T, n)``.

    Classes with a single example cannot sit on both sides and are left where
    the permutation put them. If a side has no donor with a spare example the
    class is left unrepaired.
    """
    perm = np.array(perm, copy=True)
    lab = np.array(labels_in_perm_order, copy=True)
    n = len(lab)
    if not 0 < T < n:
        return perm
    classes = sorted(set(lab.tolist()), key=str)
    for _ in range(len(classes) + 2):
        changed = False
        for label in classes:
            idx = np.flatnonzero(lab == label)
            if len(idx) < 2:
                continue
            in_pre = idx[idx < T]
            in_post = idx[idx >= T]
            if len(in_pre) == 0:
                donor = _donor(lab, 0, T, label)
                if donor is None:
                    continue
                _swap(perm, lab, int(in_post[0]), int(donor))
                changed = True
            elif len(in_post) == 0:
                donor = _donor(lab, T, n, label)
                if donor is None:
                    continue
                _swap(perm, lab, int(in_pre[0]), int(donor))
                changed = True
        if not changed:
            break
    return perm


def stationary_permutation(labels: Sequence, T: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle, then repair so classes with >= 2 points sit on both sides of T."""
    lab0 = _as_labels(labels)
    n = len(lab0)
    perm = rng.permutation(n)
    return repair_stationary(perm, lab0[perm], int(T))


def choose_held_out_class(labels: Sequence, rel_classes: Sequence, T: int) -> str:
    """Rarest relevant class that can be placed entirely after T."""
    lab = _as_labels(labels)
    n = len(lab)
    counts = Counter(lab.tolist())
    candidates = []
    for label in rel_classes:
        key = str(label)
        count = int(counts.get(key, 0))
        if count >= 1 and count <= n - T and (n - count) >= T:
            candidates.append(key)
    if not candidates:
        raise ValueError(
            f"No relevant class fits entirely after T={T} (n={n}). "
            "Reduce T or the held-out class."
        )
    candidates.sort(key=lambda label: (counts[label], label))
    return candidates[0]


def late_arrival_permutation(
    labels: Sequence,
    rel_classes: Sequence,
    T: int,
    rng: np.random.Generator,
    held_out: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Permutation that places every example of one relevant class at index >= T."""
    lab = _as_labels(labels)
    n = len(lab)
    T = int(T)
    if not 0 < T < n:
        raise ValueError(f"T must be in (0, n), got T={T}, n={n}")
    held = str(held_out) if held_out is not None else choose_held_out_class(lab, rel_classes, T)
    held_idx = np.flatnonzero(lab == held)
    other_idx = np.flatnonzero(lab != held)
    if len(held_idx) == 0:
        raise ValueError(f"held-out class {held!r} is not in the stream")
    if len(other_idx) < T:
        raise ValueError(
            f"Cannot fill a pool of {T} without class {held!r}: "
            f"only {len(other_idx)} other points"
        )
    if len(held_idx) > n - T:
        raise ValueError(
            f"Class {held!r} has {len(held_idx)} points, suffix only has {n - T}"
        )
    other_idx = rng.permutation(other_idx)
    suffix = np.concatenate([other_idx[T:], held_idx])
    suffix = rng.permutation(suffix)
    perm = np.concatenate([other_idx[:T], suffix])
    return perm.astype(int, copy=False), held


def apply_permutation(X: np.ndarray, labels: np.ndarray, relevance: np.ndarray, perm: np.ndarray):
    perm = np.asarray(perm, dtype=int)
    return X[perm], labels[perm], relevance[perm]


def assert_stationary(labels: Sequence, T: int) -> None:
    lab = _as_labels(labels)
    for label, count in Counter(lab.tolist()).items():
        if count < 2:
            continue
        in_pre = int(np.sum(lab[:T] == label))
        in_post = int(np.sum(lab[T:] == label))
        if in_pre == 0 or in_post == 0:
            raise AssertionError(f"class {label!r} missing on one side of T (pre={in_pre}, post={in_post})")


def assert_late_arrival(labels: Sequence, T: int, held_out: str) -> None:
    lab = _as_labels(labels)
    if np.any(lab[:T] == held_out):
        raise AssertionError(f"held-out class {held_out!r} appears before T")
    if not np.any(lab[T:] == held_out):
        raise AssertionError(f"held-out class {held_out!r} does not appear after T")
