# Warm-start A/RED

A short warm-up pool is labeled with farthest-first queries. Those labels seed A/RED, and every later point is streamed through A/RED.

The warm-up length is the size of that initial pool. It is a field in a JSON config. Changing it changes only how long the prefix is.

## Design

`ared/warm_start.py` is the whole method, in order:

1. Load the chosen config.
2. Take the prefix `points[:warmup]`.
3. Query that prefix farthest-first. The first query is the prefix point farthest from the prefix centroid. Each later query is the unlabeled prefix point whose nearest already-queried point is as far away as possible. Every queried index is inside the prefix.
4. Seed `ARED` with those labels. One cluster is opened per label.
5. Stream every point after the prefix through `ARED`.

`ared/ared.py` is the A/RED algorithm, class `ARED`. A streamed point is queried when its Euclidean distance to the nearest buffered point, times kappa, is greater than that point's cluster comparison distance. A nearby relevant point in the k nearest buffered points also forces a query. A query stores the point's label and may open or join a cluster.

Four parts of A/RED are swappable objects. The main sequence calls the object named in the config and does not contain the variant math:

| Folder | Config field | Choices |
| --- | --- | --- |
| `comparison_distance/` | `comparison_distance` | `diameter`, `average_nearest_neighbor` |
| `smart_forgetting/` | `smart_forgetting` | `plain` (drop the oldest point), `keep_relevant` (drop the oldest point that is not relevant) |
| `neighborhood_merge/` | `neighborhood_merge` | `disabled`, `same_label` |
| `singleton_merge/` | `singleton_merge` | `disabled`, `merge_singletons` |

`diameter` is the farthest pair in the cluster. `average_nearest_neighbor` is the average distance from each point to its nearest other point. A cluster with fewer than two points has comparison distance 0.

The command-line program builds a small deterministic stream from `stream_seed`, `stream_n`, and `stream_dim` in the config. `stream_n` must be at least `warmup`.

## Layout

```
main.py                     command-line entry
ared/warm_start.py          load, prefix, far-point, seed, stream
ared/ared.py                the A/RED algorithm
ared/far_point.py           farthest-first order on one prefix
ared/config.py              JSON load and numbered selection
ared/stream.py              the built-in synthetic stream
comparison_distance/        comparison-distance strategies
smart_forgetting/           buffer-eviction strategies
neighborhood_merge/         neighbor-merge strategies
singleton_merge/            singleton-merge strategies
configs/                    JSON configs
tests/                      pytest
```

## Run

Python 3.11 or newer, and numpy.

```
python main.py tiny.json
```

A bare file name is looked up in `configs/`. The `.json` suffix is optional, so `python main.py tiny` is the same file. A path such as `configs/tiny.json` is also accepted.

With no file name, the program lists `configs/*.json` in sorted order and reads one number from stdin:

```
python main.py
```

```
Available configs:
1. forget_keep_relevant.json
2. forget_plain.json
3. neighbor_tiny.json
4. tiny.json
5. warmup_4.json
6. warmup_8.json
Select a config number:
```

An unknown name, a number that is not on the list, or an empty selection prints an error and does not run another config.

A successful run prints four lines:

```
config: tiny.json
pool_length: 3
far_point_indices: 2,1,0
stream_query_count: 5
```

`pool_length` is the config's `warmup`. `far_point_indices` are the farthest-first queries, in order. `stream_query_count` is how many points after the pool were queried.

Shipped configs:

| File | What it changes |
| --- | --- |
| `tiny.json` | Warm-up of 3, diameter comparison distance |
| `neighbor_tiny.json` | Same stream as `tiny.json`, average nearest neighbor |
| `warmup_4.json` | Warm-up of 4 |
| `warmup_8.json` | Warm-up of 8 on the same stream as `warmup_4.json` |
| `forget_plain.json` | Buffer of 2, drop the oldest point |
| `forget_keep_relevant.json` | Buffer of 2, keep relevant points |

A config looks like this:

```json
{
  "warmup": 3,
  "kappa": 1.0,
  "buffer_size": 30,
  "k_neighbors": 2,
  "comparison_distance": "diameter",
  "smart_forgetting": "plain",
  "neighborhood_merge": "disabled",
  "singleton_merge": "disabled",
  "stream_seed": 0,
  "stream_n": 8,
  "stream_dim": 2
}
```

## Tests

```
python -m pip install pytest
python -m pytest
```
