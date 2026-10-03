"""Dataset loaders.

NICE is the synthetic stream from the vendored generator (or a smaller draw
with the same class-size schedule, for CPU runs). PARKING_LOT_DINO reads the
cached 16-d latents. DINOv2 and the 768-to-16 autoencoder run only when that
cache is absent and the source images are present. EMNIST_DINO / MNIST_DINO
load only when an embedding cache is already on disk.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.vendor_bootstrap import PROJECT_ROOT, VENDOR_ARED, bootstrap_ared

PARKING_LATENT_NAME = "PARKING_LOT_DINO_latents_16d.npy"
PARKING_LABELS_NAME = "labels.csv"


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    labels: np.ndarray
    relevance: np.ndarray
    rel_classes: List[str]
    sparsity_levels: List[Tuple[str, float]]


def relevance_from_labels(labels: Sequence, n_rel_classes: int):
    """Mark the ``n_rel_classes`` rarest labels as relevant.

    Ties break toward the label that sorts later, which matches the vendored
    parking-lot rule of taking the tail of a most-common-first list.
    """
    lab = np.asarray([str(v) for v in labels], dtype=object)
    if n_rel_classes < 0:
        raise ValueError(f"n_rel_classes must be non-negative, got {n_rel_classes}")
    counts = Counter(lab.tolist())
    most_common = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    total = max(len(lab), 1)
    sparsity = [(label, count / total) for label, count in most_common]
    n_rel = min(int(n_rel_classes), len(most_common))
    rel_classes = [label for label, _ in most_common[-n_rel:]] if n_rel else []
    rel_set = set(rel_classes)
    relevance = np.asarray([label in rel_set for label in lab.tolist()], dtype=bool)
    return lab, relevance, rel_classes, sparsity


def _nice_centers(num_classes: int, seed: int, min_distance: float = 10.0) -> np.ndarray:
    rs = np.random.RandomState(seed)
    centers = [rs.uniform(-20.0, 20.0, size=2)]
    guard = 0
    while len(centers) < num_classes:
        guard += 1
        if guard > 100000:
            raise RuntimeError("failed to place separated NICE centers")
        cand = rs.uniform(-20.0, 20.0, size=2)
        if all(float(np.linalg.norm(cand - c)) >= min_distance for c in centers):
            centers.append(cand)
    return np.vstack(centers)


def generate_scaled_nice(
    n_rel_classes: int,
    seed: int = 0,
    base_count: int = 800,
    min_count: int = 20,
    num_classes: int = 10,
    cluster_std: float = 1.0,
    verbose: bool = False,
) -> DatasetBundle:
    """Same schedule as the vendored NICE generator, with a smaller base count.

    Vendored sizes are ``500_000 // 2**i``. ``base_count`` replaces 500_000 and
    ``min_count`` stops the tail from collapsing to one (or zero) points, which
    a pool/watch split cannot otherwise place on both sides of T.
    """
    from sklearn.datasets import make_blobs

    if base_count <= 0:
        raise ValueError("base_count must be positive; use load_nice_full for the vendored sizes")
    samples = [max(int(min_count), int(base_count) // (2 ** i)) for i in range(num_classes)]
    centers = _nice_centers(num_classes, int(seed))
    X, y = make_blobs(
        n_samples=samples,
        centers=centers,
        cluster_std=float(cluster_std),
        random_state=int(seed),
    )
    rs = np.random.RandomState(int(seed) + 1)
    order = rs.permutation(len(y))
    X = np.asarray(X[order], dtype=np.float64)
    y = np.asarray(y[order])
    labels, relevance, rel_classes, sparsity = relevance_from_labels(y.astype(str), n_rel_classes)
    if verbose:
        print(f"NICE scaled samples per class: {samples} (n={len(X)})")
        print(f"Relevant classes (rarest {n_rel_classes}): {rel_classes}")
    return DatasetBundle(
        name="NICE",
        X=X,
        labels=labels,
        relevance=relevance,
        rel_classes=rel_classes,
        sparsity_levels=sparsity,
    )


def load_nice_full(n_rel_classes: int, seed: int = 0, verbose: bool = False) -> DatasetBundle:
    """Vendored NICE stream (~1e6 points, 10 well-separated Gaussians)."""
    bootstrap_ared()
    from NICE_Data_Processing import generate_synthetic_dataset_with_relevance

    X, y_w_rel, sparsity_levels, rel_classes = generate_synthetic_dataset_with_relevance(
        int(n_rel_classes), int(seed)
    )
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray([str(pair[0]) for pair in y_w_rel], dtype=object)
    # Recompute relevance so the rarest-n rule is the one used downstream even
    # if the helper's printout and our tie break ever diverge. For the
    # unfloored schedule the rarest classes are the last n integer labels.
    labels, relevance, rel_classes, sparsity = relevance_from_labels(labels, n_rel_classes)
    if verbose:
        print(f"NICE full stream: n={len(X)} relevant={rel_classes}")
    del sparsity_levels
    return DatasetBundle("NICE", X, labels, relevance, rel_classes, sparsity)


def _unique_dirs(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def parking_search_dirs() -> List[Path]:
    """Directories that may hold PARKING_LOT_DINO_latents_16d.npy and labels.csv."""
    dirs: List[Path] = []
    env = os.environ.get("PARKING_LOT_DATA_DIR")
    if env:
        dirs.append(Path(env))
    roots = [Path.cwd(), PROJECT_ROOT, PROJECT_ROOT / "data", VENDOR_ARED]
    for root in roots:
        dirs.append(root / "Datasets" / "Parking_Lot_Data")
        dirs.append(root / "Parking_Lot_Data")
    return _unique_dirs(dirs)


def find_parking_latents() -> Optional[Tuple[Path, Path]]:
    for directory in parking_search_dirs():
        features = directory / PARKING_LATENT_NAME
        labels = directory / PARKING_LABELS_NAME
        if features.is_file() and labels.is_file():
            return features, labels
    return None


def _parking_from_files(features: Path, labels_path: Path, n_rel_classes: int, verbose: bool) -> DatasetBundle:
    import pandas as pd

    X = np.load(features).astype(np.float64)
    if X.ndim != 2:
        raise ValueError(f"{features} must be a 2-d latent matrix, got {X.shape}")
    frame = pd.read_csv(labels_path)
    if "label" not in frame.columns:
        raise ValueError(f"{labels_path} has no 'label' column (columns={list(frame.columns)})")
    raw = frame["label"].astype(str).to_numpy()
    if len(raw) != len(X):
        raise ValueError(
            f"Feature/label length mismatch: X has {len(X)} rows, labels have {len(raw)} ({features})"
        )
    labels, relevance, rel_classes, sparsity = relevance_from_labels(raw, n_rel_classes)
    if verbose:
        print(f"PARKING_LOT_DINO latents {X.shape} from {features}")
        print(f"Relevant classes (rarest {n_rel_classes}): {rel_classes}")
    return DatasetBundle("PARKING_LOT_DINO", X, labels, relevance, rel_classes, sparsity)


def _try_extract_parking_latents(verbose: bool) -> Optional[Tuple[Path, Path]]:
    """Run the vendored DINOv2 + autoencoder path only if source images exist."""
    source_dirs = []
    for directory in parking_search_dirs():
        if (directory / "features.pkl").is_file() or (directory / "resized_features_224.pkl").is_file():
            source_dirs.append(directory)
    if not source_dirs:
        return None
    bootstrap_ared()
    # The preprocessor reads Datasets/Parking_Lot_Data/* relative to the
    # process cwd and writes the 16-d cache into out_dir.
    source = source_dirs[0]
    cwd = Path.cwd()
    try:
        # If the pickle already lives under Datasets/Parking_Lot_Data, run
        # from that tree's grandparent when the layout matches the vendor.
        if source.name == "Parking_Lot_Data" and source.parent.name == "Datasets":
            os.chdir(source.parent.parent)
            out_dir = "Datasets/Parking_Lot_Data"
        else:
            os.chdir(source)
            out_dir = "."
        if verbose:
            print(f"Latent cache missing. Running DINOv2 autoencoder from {Path.cwd()} ...")
        from DINOv2_Parking_Lot import run_dinov2_autoencoder_preprocessing

        run_dinov2_autoencoder_preprocessing(out_dir=out_dir, do_viz=False)
    finally:
        os.chdir(cwd)
    return find_parking_latents()


def load_parking_lot_dino(n_rel_classes: int, seed: int = 0, verbose: bool = False) -> DatasetBundle:
    """16-d parking-lot latents. ``seed`` is accepted for API symmetry and unused."""
    del seed
    found = find_parking_latents()
    if found is None:
        found = _try_extract_parking_latents(verbose)
    if found is None:
        searched = "\n".join(f"  - {p}" for p in parking_search_dirs())
        raise FileNotFoundError(
            "PARKING_LOT_DINO latents were not found, and no source image pickle "
            "was available to run the vendored DINOv2 path.\n"
            f"Place {PARKING_LATENT_NAME} and {PARKING_LABELS_NAME} in one of:\n{searched}\n"
            "or set PARKING_LOT_DATA_DIR. Expected vendor layout: "
            "vendor/A_RED_INF/Datasets/Parking_Lot_Data/."
        )
    return _parking_from_files(found[0], found[1], n_rel_classes, verbose)


def _search_files(names: Sequence[str]) -> Optional[Path]:
    roots = [Path.cwd(), PROJECT_ROOT, PROJECT_ROOT / "data", VENDOR_ARED, VENDOR_ARED / "Datasets"]
    for root in roots:
        for name in names:
            for path in root.rglob(name):
                if path.is_file():
                    return path
    return None


def _bundle_from_xy(name: str, X, raw_labels, n_rel_classes: int, verbose: bool) -> DatasetBundle:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} features must be 2-d, got {X.shape}")
    if len(raw_labels) != len(X):
        raise ValueError(f"{name} feature/label length mismatch: {len(X)} vs {len(raw_labels)}")
    labels, relevance, rel_classes, sparsity = relevance_from_labels(raw_labels, n_rel_classes)
    if verbose:
        print(f"{name} embeddings {X.shape}; relevant={rel_classes}")
    return DatasetBundle(name, X, labels, relevance, rel_classes, sparsity)


def _labels_from_cached_y(y_obj):
    if isinstance(y_obj, dict) and "labels" in y_obj:
        return list(y_obj["labels"])
    arr = np.asarray(y_obj, dtype=object)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        return [str(v) for v in arr[:, 0].tolist()]
    if arr.ndim == 1 and len(arr) and isinstance(arr[0], (tuple, list)):
        return [str(v[0]) for v in arr.tolist()]
    return [str(v) for v in arr.tolist()]


def load_emnist_dino(n_rel_classes: int, seed: int = 0, verbose: bool = False) -> DatasetBundle:
    """Load cached EMNIST embeddings if they already exist. Does not extract."""
    del seed
    feat = _search_files(("emnist_dino_features.pkl", "EMNIST_DINO_latents.npy", "emnist_dino_latents.npy"))
    if feat is None:
        raise FileNotFoundError(
            "EMNIST_DINO embeddings were not found. This project does not "
            "re-run DINOv2 on EMNIST. Place emnist_dino_features.pkl (and "
            "emnist_dino_y.pkl) under vendor/A_RED_INF/Datasets/EMNIST/."
        )
    if feat.suffix == ".npy":
        X = np.load(feat)
        y_path = feat.with_name("emnist_dino_y.pkl")
        if not y_path.is_file():
            raise FileNotFoundError(f"Found {feat} but not {y_path}")
        with y_path.open("rb") as handle:
            raw = _labels_from_cached_y(pickle.load(handle))
    else:
        with feat.open("rb") as handle:
            X = pickle.load(handle)
        y_path = feat.with_name("emnist_dino_y.pkl")
        if not y_path.is_file():
            raise FileNotFoundError(f"Found {feat} but not {y_path}")
        with y_path.open("rb") as handle:
            raw = _labels_from_cached_y(pickle.load(handle))
    return _bundle_from_xy("EMNIST_DINO", X, raw, n_rel_classes, verbose)


def load_mnist_dino(n_rel_classes: int, seed: int = 0, verbose: bool = False) -> DatasetBundle:
    """Load cached MNIST embeddings if present. Raw pixels are not used."""
    del seed
    feat = _search_files(("mnist_dino_features.pkl", "MNIST_DINO_latents.npy", "mnist_dino_latents.npy"))
    if feat is None:
        raise FileNotFoundError(
            "MNIST_DINO embeddings were not found. Raw-pixel MNIST is not a "
            "supported representation. Place a DINOv2 embedding cache named "
            "mnist_dino_features.pkl or MNIST_DINO_latents.npy on disk."
        )
    if feat.suffix == ".npy":
        X = np.load(feat)
        y_path = feat.with_name(feat.name.replace("latents", "y").replace("features", "y"))
        candidates = list(feat.parent.glob("*y*.pkl")) + list(feat.parent.glob("*label*"))
        if not candidates:
            raise FileNotFoundError(f"Found {feat} but no label file beside it")
        y_path = candidates[0]
        with open(y_path, "rb") as handle:
            raw = _labels_from_cached_y(pickle.load(handle))
    else:
        with feat.open("rb") as handle:
            X = pickle.load(handle)
        y_candidates = list(feat.parent.glob("mnist_dino_y.pkl"))
        if not y_candidates:
            raise FileNotFoundError(f"Found {feat} but not mnist_dino_y.pkl")
        with y_candidates[0].open("rb") as handle:
            raw = _labels_from_cached_y(pickle.load(handle))
    return _bundle_from_xy("MNIST_DINO", X, raw, n_rel_classes, verbose)


def load_dataset(
    name: str,
    n_rel_classes: int,
    seed: int = 0,
    nice_base: int = 800,
    nice_min_count: int = 20,
    verbose: bool = False,
) -> DatasetBundle:
    """Load a named stream.

    ``nice_base=0`` selects the vendored full NICE generator. Any positive
    ``nice_base`` builds the same 10-class Gaussian family at that scale.
    """
    key = name.strip().upper()
    if key == "NICE":
        if int(nice_base) == 0:
            if verbose:
                print("Loading full vendored NICE stream (about 1e6 points).")
            return load_nice_full(n_rel_classes, seed=seed, verbose=verbose)
        return generate_scaled_nice(
            n_rel_classes,
            seed=seed,
            base_count=int(nice_base),
            min_count=int(nice_min_count),
            verbose=verbose,
        )
    if key == "PARKING_LOT_DINO":
        return load_parking_lot_dino(n_rel_classes, seed=seed, verbose=verbose)
    if key == "EMNIST_DINO":
        return load_emnist_dino(n_rel_classes, seed=seed, verbose=verbose)
    if key in {"MNIST_DINO", "MNIST"}:
        return load_mnist_dino(n_rel_classes, seed=seed, verbose=verbose)
    raise ValueError(
        f"Unknown dataset {name!r}. Expected NICE, PARKING_LOT_DINO, EMNIST_DINO, or MNIST_DINO."
    )
