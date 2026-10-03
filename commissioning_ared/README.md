# Commissioning, then streaming A/RED

Pool-phase labeling on a prefix `X[0:T]`, then a streaming watch phase on
`X[T:N]`. The commissioning query rule is global farthest-first. The watch
phase is the vendored streaming detector, unchanged: a point is queried when
`distance * kappa > comp_distance` (or when a nearby buffered point is relevant).

Four methods, compared after `T` at the same budget `B0` and the same `kappa`:

| Method | Pool phase | Watch phase |
|---|---|---|
| `cold_ared` | none (detector runs from index 0; `B0` is only the shared x-axis) | continue from `T` |
| `random_then_ared` | `B0` uniform labels on `X[0:T]` | seeded detector |
| `farpoint_then_ared` | global farthest-first, budget `B0` | seeded detector |
| `oracle_then_ared` | one label per pool class, rarest first, then random leftovers | seeded detector |

Running farthest-first on the whole pool is not a fifth method. Figures draw a
dashed line at the fraction of post-`T` relevant mass whose class already
appears in the pool. The watch phase can still exceed that line by discovering
a class that was absent from the pool.

## Layout

```
src/farpoint_fft.py     global farthest-first commissioner (default) and the legacy per-cluster rule
src/handoff.py          CommissioningState -> streaming detector
src/protocols.py        the four methods
src/metrics.py          query precision and relevant recall
src/datasets.py         NICE and PARKING_LOT_DINO loaders
src/run_experiment.py   one run
src/run_sweep.py        YAML grid
vendor/farpoint/        unmodified reference implementation
vendor/A_RED_INF/       unmodified streaming detector
```

`QUERY_RULE` is `global_fft` (default) or `legacy_cluster_fp`.

Global rule, with `L` the labeled pool indices and `U` the unlabeled ones:

```
next = argmax_{u in U}  min_{l in L}  ||X[u] - X[l]||_2
```

The first query is the pool point farthest from the pool centroid
(`--first-query random` draws uniformly instead). A new label opens one
cluster. A repeated label is appended to that class. Unlabeled pool points can
be assigned to the nearest labeled point for prediction; that assignment does
not choose the next query. The legacy rule keeps the per-cluster far point and,
on a label mismatch, partitions that cluster by nearest labeled point. It does
not run a constrained k-means or a density split.

## Metrics

Watch-phase numbers use the vendored definitions, on the confusion matrix and
query counters accumulated after `T` only.

* **Query precision** = `num_correct_queries / num_queries`. Inside the detector a query counts as correct when the queried point is relevant.
* **Relevant recall** = queried relevant points / streamed relevant points, from `calculate_single_rel_recall` (relevant-class diagonal over the relevant-class row sum).

Also logged, per run, as JSONL:

* commissioning queries, classes discovered, relevant classes discovered
* discovery latency: stream index of the first query of each relevant class (pool queries included)
* queries whose label was already known to be irrelevant
* missed relevant mass after `T`
* total labels, labels per relevant example found, post-`T` query rate

## Data

Representation is fixed. Do not refit DINOv2 or the 768-to-16 map after `T`.
The loader reads a cache and only calls the vendored DINOv2 path when the
cache is absent and source images are already present.

`PARKING_LOT_DINO` is accepted from any of:

```
$PARKING_LOT_DATA_DIR/PARKING_LOT_DINO_latents_16d.npy
$PARKING_LOT_DATA_DIR/labels.csv
vendor/A_RED_INF/Datasets/Parking_Lot_Data/PARKING_LOT_DINO_latents_16d.npy
vendor/A_RED_INF/Datasets/Parking_Lot_Data/labels.csv
Datasets/Parking_Lot_Data/   (relative to the working directory or this project)
Parking_Lot_Data/
```

`labels.csv` must have a `label` column. Relevance is the `N_REL_CLASSES`
rarest labels (parking-lot settings in the sweep: 6 and 8).

`NICE` is the vendored 10-class Gaussian family. `--nice-base 0` calls that
generator (~1e6 points). A positive `--nice-base` uses the same
`base // 2**i` schedule with a floor of `--nice-min-count`, which is what
`make toy` runs. `EMNIST_DINO` and `MNIST_DINO` load only when an embedding
cache is already on disk. Raw pixels are not a supported representation.

Stream constructions are index permutations of one array:

* `stationary`: every class with at least two examples appears on both sides of `T`
* `late_arrival`: every example of the rarest relevant class that fits is held until after `T`

## Setup

Python 3.10 or 3.11. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

`make env`, `make test`, `make toy`, `make parking`, and `make sweep` wrap the
same steps when GNU make is available. Torch is required only if the parking-lot
latent cache is missing and source images have to be embedded. NICE does not
import it.

The legacy far-point flag does not need `active-semi-supervised-clustering`.

## One run

```powershell
python -m src.run_experiment --method farpoint_then_ared `
    --data PARKING_LOT_DINO --t-frac 0.1 --B0 50 --kappa 0.5 `
    --query-rule global_fft --seed 0 --out results/run.jsonl
```

Prints commissioning discoveries and watch-phase query precision / relevant recall.

```powershell
python -m src.run_experiment --method cold_ared --data NICE --stream stationary `
    --t-frac 0.2 --B0 20 --kappa 1.0 --seed 0 --n-rel-classes 4 `
    --l-buf-size 200 --nice-base 400 --out results/toy.jsonl --overwrite
```

Defaults when the flag is omitted: parking-lot kappa `0.5` and 6 relevant
classes; NICE kappa `1.0` and 4 relevant classes; buffer `1000`; comparison
distance is average nearest neighbor (`--qs-var 1`); `k = 2`.

## Matrix

```powershell
python -m src.run_sweep --config experiments/configs/nice_small.yaml
python -m src.run_sweep --config experiments/configs/nice_matrix.yaml --out results/nice_matrix
python -m src.run_sweep --config experiments/configs/parking_lot.yaml --out results/parking_lot
python -m src.plotting --jsonl results/nice_small/runs.jsonl --out results/figures --summary results/summary.csv
```

`nice_small.yaml` is the short driver (`make sweep`): four methods, both
stream constructions, one seed. `nice_matrix.yaml` and `parking_lot.yaml` are
the full grids (`T` fraction 0.10 and 0.20, budgets 20/50/100, three kappas,
two buffer sizes, five seeds, both streams). Cold start is executed once per
setting and copied across budgets, because it does not read `B0`.

Figures, written under the sweep's `figures/` directory:

1. `fig1_watch_rr_vs_b0_stationary.png` — watch-phase relevant recall vs `B0`
2. `fig2_watch_rr_vs_b0_late_arrival.png` — same, late arrival
3. `fig3_qp_vs_rr_kappa.png` — query precision vs relevant recall across kappa
4. `fig4_discovery_latency.png` — stream index of the first query of each relevant class

`make toy` runs cold vs farthest-first on scaled NICE (budget 20, one seed,
both stream constructions) and writes `results/toy.jsonl`, `results/summary.csv`,
and `results/figures/`. `make parking` runs the smoke config and exits with the
searched paths if `PARKING_LOT_DINO_latents_16d.npy` is not on disk.
