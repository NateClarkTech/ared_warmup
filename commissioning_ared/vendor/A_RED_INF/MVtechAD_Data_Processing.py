import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle


# ──────────────────────────────────────────────────────────────────────────────
# MVtec AD dataset structure (per object category):
#
#   MVtechAD/
#     <object>/
#       train/
#         good/          ← only normal samples used for training
#       test/
#         good/          ← normal test samples   → label "normal_<object>",  NOT relevant
#         <defect_type>/ ← anomalous test samples → label "<defect_type>_<object>", relevant
#       ground_truth/
#         <defect_type>/ ← segmentation masks (not used here)
#
# Both train/good and test/good are loaded as normal samples.
# All test/<defect_type> folders (defect_type != "good") are loaded as anomalies.
#
# label_str   = "<defect_type>_<object>"  (anomalous, defect_type = folder name)
#             = "normal_<object>"          (normal, from train/good or test/good)
# is_relevant = True for anomalous samples, False for normal
# ──────────────────────────────────────────────────────────────────────────────

MVTECHAD_OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

IMG_SIZE = (256, 256)   # resize target; tune to your memory / compute budget


def _load_image_flat(img_path: str, img_size: tuple[int, int]) -> np.ndarray | None:
    """Load an image, resize, convert to RGB, and flatten to a 1-D float32 array."""
    try:
        img = Image.open(img_path).convert("L").resize(img_size, Image.LANCZOS)
        return np.array(img, dtype=np.float32).flatten() / 255.0
    except Exception as e:
        print(f"  [WARN] Could not load {img_path}: {e}")
        return None


def load_mvtechad_category(
    mvtechad_root: str,
    object_name: str,
    include_train: bool = True,
    img_size: tuple[int, int] = IMG_SIZE,
    verbose: bool = False,
) -> tuple[np.ndarray, list[tuple[str, bool]]]:
    """
    Load all samples for one MVtec AD object category.

    Parameters
    ----------
    mvtechad_root : path to the top-level MVtechAD/ directory
    object_name   : e.g. 'bottle', 'carpet', 'wood', …
    include_train : if True, also load train/good/ as normal samples
    img_size      : (width, height) to resize images to before flattening
    verbose       : print per-category statistics

    Returns
    -------
    X        : np.ndarray, shape (n_samples, img_size[0]*img_size[1])
    y_w_rel  : list of (label_str, is_relevant_bool)
               label_str  = "<defect_type>_<object>"  or  "normal_<object>"
               is_relevant = True for anomalous samples
    """
    obj_dir = os.path.join(mvtechad_root, object_name)
    if not os.path.isdir(obj_dir):
        seen_root = os.path.abspath(mvtechad_root) if os.path.isdir(mvtechad_root) else None
        if seen_root:
            seen_entries = sorted(os.listdir(seen_root))
            seen_str = "\n  ".join(seen_entries) if seen_entries else "(empty)"
        else:
            cwd_entries = sorted(os.listdir("."))
            seen_str = "\n  ".join(cwd_entries) if cwd_entries else "(empty)"
            seen_root = f"{os.path.abspath('.')}  (mvtechad_root '{mvtechad_root}' not found, showing cwd)"
        raise FileNotFoundError(
            f"Object directory not found: {os.path.abspath(obj_dir)}\n"
            f"Expected MVtechAD structure: {mvtechad_root}/<object>/test/<label>/\n"
            f"Contents of {seen_root}:\n  {seen_str}"
        )

    X_list:  list[np.ndarray]         = []
    y_w_rel: list[tuple[str, bool]]   = []

    # ── Helper: walk one folder and append samples ───────────────────────────
    def _ingest_folder(folder: str, label_str: str, is_anomaly: bool) -> None:
        if not os.path.isdir(folder):
            return
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            pixels = _load_image_flat(os.path.join(folder, fname), img_size)
            if pixels is not None:
                X_list.append(pixels)
                y_w_rel.append((label_str, is_anomaly))

    # ── train/good → normal samples ──────────────────────────────────────────
    if include_train:
        train_good = os.path.join(obj_dir, "train", "good")
        _ingest_folder(train_good, f"normal_{object_name}", is_anomaly=False)

    # ── test/<label> folders ─────────────────────────────────────────────────
    test_dir = os.path.join(obj_dir, "test")
    if not os.path.isdir(test_dir):
        print(f"  [WARN] No test/ directory found for {object_name}, skipping.")
    else:
        for label_folder in sorted(os.listdir(test_dir)):
            label_path = os.path.join(test_dir, label_folder)
            if not os.path.isdir(label_path):
                continue

            if label_folder.lower() == "good":
                # Normal test samples
                _ingest_folder(label_path, f"normal_{object_name}", is_anomaly=False)
            else:
                # Anomalous test samples — label = folder name
                _ingest_folder(label_path, f"{label_folder}_{object_name}", is_anomaly=True)

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, img_size[0] * img_size[1] * 3))

    if verbose:
        counts = Counter(lbl for lbl, _ in y_w_rel)
        n_anom = sum(1 for _, r in y_w_rel if r)
        print(f"  [{object_name}]  {len(y_w_rel)} samples  |  "
              f"{n_anom} anomalous  |  {len(counts)} unique labels")
        for lbl, cnt in counts.most_common():
            print(f"      {lbl}: {cnt}")

    return X, y_w_rel



def _shuffle_within_categories(X, y_w_rel, category_separator="_", seed=42):
    """
    Shuffle X and y_w_rel so that samples within each object category are
    randomized, but categories themselves stay grouped in their original order.

    The object category is the suffix after the last '_' in the label string,
    e.g. "normal_candle" and "burned_candle" both belong to category "candle".
    """
    import random
    rng = random.Random(seed)

    # Build ordered list of unique categories (preserving first-seen order)
    seen = []
    seen_set = set()
    for lbl, _ in y_w_rel:
        # category = everything after the first underscore
        cat = lbl.split("_", 1)[-1]
        if cat not in seen_set:
            seen.append(cat)
            seen_set.add(cat)

    # Group indices by category
    from collections import defaultdict
    cat_indices = defaultdict(list)
    for i, (lbl, _) in enumerate(y_w_rel):
        cat = lbl.split("_", 1)[-1]
        cat_indices[cat].append(i)

    # Shuffle within each category, then concatenate in original category order
    final_order = []
    for cat in seen:
        idxs = cat_indices[cat]
        rng.shuffle(idxs)
        final_order.extend(idxs)

    X_out     = X[final_order]
    y_out     = [y_w_rel[i] for i in final_order]
    return X_out, y_out


def mvtechad_setup_for_main(
    N_REL_CLASSES,
    VERBOSE_FLAGS,
    seed,
    mvtechad_root: str = None,                    # ← Set to None for auto-detection
    object_categories: list[str] | None = None,
    include_train: bool = True,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple[np.ndarray, list[tuple[str, bool]], list[tuple[str, float]], list[str]]:
    """
    Load the full MVtec AD dataset with automatic path detection for both Nate and Rohan.
    Cache files are now stored in ~/.cache/mvtechad/ (cleaner, avoids Dropbox sync).
    """
    verbose = 0 in VERBOSE_FLAGS

    # ── Smart Dataset Path Detection ───────────────────────────────────────
    if mvtechad_root is None:
        nate_path = "Datasets/MVtechAD"
        rohan_path = os.path.expanduser("~/Dropbox/Research/Datasets/mvtec")

        if os.path.isdir(nate_path):
            mvtechad_root = nate_path
            if verbose:
                print(f"✅ Using Nate's path: {mvtechad_root}")
        elif os.path.isdir(rohan_path):
            mvtechad_root = rohan_path
            if verbose:
                print(f"✅ Using Rohan's path: {mvtechad_root}")
        else:
            raise FileNotFoundError(
                "MVtec AD dataset not found in any default location.\n"
                f"Checked:\n"
                f"  • Nate:  {os.path.abspath(nate_path)}\n"
                f"  • Rohan: {rohan_path}\n\n"
                "Please pass the correct path manually:\n"
                "mvtechad_setup_for_main(mvtechad_root='/your/path')"
            )

    # ── Cache Location (Improved) ──────────────────────────────────────────
    cache_dir = os.path.expanduser("~/.cache/mvtechad")
    os.makedirs(cache_dir, exist_ok=True)

    cache_X = os.path.join(cache_dir, "mvtechad_processed_X.npz")
    cache_y = os.path.join(cache_dir, "mvtechad_processed_y.pkl")

    if verbose:
        print(f"Cache directory: {cache_dir}")

    # ── Load from cache if available ───────────────────────────────────────
    if os.path.exists(cache_X) and os.path.exists(cache_y):
        if verbose:
            print(f"Loading MVtechAD dataset from cache: {cache_X}")
        X = np.load(cache_X)["X"]
        with open(cache_y, "rb") as f:
            y_w_rel = pickle.load(f)
    else:
        # ── Load from raw dataset ──────────────────────────────────────────
        if object_categories is None:
            object_categories = MVTECHAD_OBJECT_CATEGORIES

        all_X: list[np.ndarray] = []
        all_y_w_rel: list[tuple[str, bool]] = []

        for obj in object_categories:
            if verbose:
                print(f"Loading MVtechAD category: {obj} …")
            X_cat, y_cat = load_mvtechad_category(
                mvtechad_root, obj,
                include_train=include_train,
                img_size=img_size,
                verbose=verbose,
            )
            all_X.append(X_cat)
            all_y_w_rel.extend(y_cat)

        X = np.concatenate(all_X, axis=0) if all_X else np.empty((0,))
        y_w_rel = all_y_w_rel

        if verbose:
            print(f"Saving cache to: {cache_dir}")
        np.savez_compressed(cache_X, X=X)
        with open(cache_y, "wb") as f:
            pickle.dump(y_w_rel, f)

    # ── Shuffle within categories ──────────────────────────────────────────
    X, y_w_rel = _shuffle_within_categories(X, y_w_rel, seed=42)

    # ── Compute sparsity levels ────────────────────────────────────────────
    labels_only = [lbl for lbl, _ in y_w_rel]
    label_counts = Counter(labels_only)
    total = len(labels_only)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in label_counts.most_common()]

    # ── Relevant labels (anomalous ones) ───────────────────────────────────
    label_relevance: dict[str, bool] = {}
    for lbl, is_rel in y_w_rel:
        label_relevance[lbl] = label_relevance.get(lbl, False) or is_rel
    relevant_labels = sorted(lbl for lbl, rel in label_relevance.items() if rel)

    # ── Final verbose summary ──────────────────────────────────────────────
    if verbose:
        print(f"\nMVtechAD dataset loaded successfully from: {mvtechad_root}")
        print(f"  X shape        : {X.shape}")
        print(f"  Total samples  : {total}")
        print(f"  Total classes  : {len(label_counts)}")
        print(f"  Relevant labels: {len(relevant_labels)}")
        n_rel = sum(1 for _, r in y_w_rel if r)
        print(f"  Anomalous      : {n_rel} ({n_rel/total*100:.1f}%)")
        print(f"  Normal         : {total - n_rel} ({(total-n_rel)/total*100:.1f}%)")

    return X, y_w_rel, sparsity_levels, relevant_labels
# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MVTECHAD_ROOT = "Datasets/MVtechAD"   # ← adjust to your local path

    X, y_w_rel, sparsity_levels, relevant_labels = mvtechad_setup_for_main(
        N_REL_CLASSES=None,
        VERBOSE_FLAGS=[0],
        seed=42,
        mvtechad_root=MVTECHAD_ROOT,
        include_train=True,
    )

    total     = len(y_w_rel)
    n_anomaly = sum(1 for _, r in y_w_rel if r)

    print("\n── Summary ──────────────────────────────────────")
    print(f"X shape        : {X.shape}")
    print(f"Total samples  : {total}")
    print(f"Anomaly rate   : {n_anomaly/total*100:.2f}%  ({n_anomaly} anomalous / {total - n_anomaly} normal)")
    print()
    for lbl, prop in sparsity_levels:
        print(f"  {lbl:<45} {prop*100:.2f}%")