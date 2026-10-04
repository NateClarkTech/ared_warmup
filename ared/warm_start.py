"""Warm start around the A/RED algorithm in ``ared/ared.py``.

Load the chosen config, take the warm-up prefix, query that prefix farthest-first,
seed A/RED with those labels, then stream every later point through A/RED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ared.config import load_config
from ared.ared import ARED
from ared.far_point import far_point_indices
from ared.stream import synthetic_stream


@dataclass(frozen=True)
class Result:
    pool_length: int
    far_point_indices: list[int]
    stream_query_count: int
    config_name: str


def run(config_name, stdin=None, points=None, labels=None, relevance=None) -> Result:
    config = load_config(config_name, stdin=stdin)
    points, labels, relevance = _stream(config, points, labels, relevance)

    prefix = points[: config.warmup]
    far_indices = far_point_indices(prefix)

    ared = ARED.from_config(config, points, labels, relevance)
    ared.seed(far_indices)

    stream_query_count = 0
    for index in range(config.warmup, len(points)):
        if ared.stream_point(index):
            stream_query_count += 1

    return Result(
        pool_length=config.warmup,
        far_point_indices=list(far_indices),
        stream_query_count=stream_query_count,
        config_name=config.name,
    )


def format_result(result: Result) -> str:
    indices = ",".join(str(index) for index in result.far_point_indices)
    return (
        f"config: {result.config_name}\n"
        f"pool_length: {result.pool_length}\n"
        f"far_point_indices: {indices}\n"
        f"stream_query_count: {result.stream_query_count}\n"
    )


def _stream(config, points, labels, relevance):
    if points is None:
        points, labels, relevance = synthetic_stream(config.stream_seed, config.stream_n, config.stream_dim)
    else:
        points = np.asarray(points, dtype=np.float64)
        labels = [str(value) for value in labels]
        relevance = [bool(value) for value in relevance]
        if points.ndim != 2:
            raise ValueError(f"points must be a 2-d array, got shape {points.shape}")
        if not (len(points) == len(labels) == len(relevance)):
            raise ValueError("points, labels, and relevance must have the same length")
    if len(points) < config.warmup:
        raise ValueError(
            f"stream length {len(points)} is shorter than warm-up {config.warmup}"
        )
    return points, labels, relevance
