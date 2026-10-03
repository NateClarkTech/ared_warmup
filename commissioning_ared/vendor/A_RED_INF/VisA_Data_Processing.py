import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# VisA dataset structure (per object category):
#
#   VisA/
#     <object>/
#       Data/
#         Images/
#           Normal/      ← normal samples
#           Anomaly/     ← anomalous samples
#         Masks/
#           Anomaly/
#       image_anno.csv   ← columns: image, label, mask
#                           label is 'normal' or comma-separated anomaly type(s)
#
# label_str  = "<first_anomaly_type>_<object>"   (anomalous)
#            = "normal_<object>"                  (normal)
# is_relevant = True  if anomalous, False if normal
# ──────────────────────────────────────────────────────────────────────────────

VISA_OBJECT_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
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


def _parse_first_label(raw_label: str) -> str:
    """Return the first (primary) anomaly type from a potentially comma-separated label."""
    return raw_label.split(",")[0].strip()


def load_visa_category(
    visa_root: str,
    object_name: str,
    img_size: tuple[int, int] = IMG_SIZE,
    verbose: bool = False,
) -> tuple[np.ndarray, list[tuple[str, bool]]]:
    """
    Load all samples for one VisA object category.

    Parameters
    ----------
    visa_root   : path to the top-level VisA/ directory
    object_name : e.g. 'candle', 'capsules', …
    img_size    : (width, height) to resize images to before flattening
    verbose     : print per-category statistics

    Returns
    -------
    X        : np.ndarray, shape (n_samples, img_size[0]*img_size[1])
    y_w_rel  : list of (label_str, is_relevant_bool)
               label_str  = "<primary_anomaly_type>_<object>"  or  "normal_<object>"
               is_relevant = True for anomalous samples
    """
    anno_path = os.path.join(visa_root, object_name, "image_anno.csv")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation CSV not found: {anno_path}\n"
            f"Expected VisA structure: {visa_root}/<object>/image_anno.csv"
        )

    df = pd.read_csv(anno_path)
    # 'image' column stores paths relative to visa_root, e.g.
    # "candle/Data/Images/Normal/0000.JPG"

    X_list: list[np.ndarray] = []
    y_w_rel: list[tuple[str, bool]] = []

    for _, row in df.iterrows():
        rel_img_path = row["image"]           # relative to visa_root
        raw_label    = str(row["label"])
        abs_img_path = os.path.join(visa_root, rel_img_path)

        pixels = _load_image_flat(abs_img_path, img_size)
        if pixels is None:
            continue

        is_anomaly   = (raw_label.lower() != "normal")
        primary_type = _parse_first_label(raw_label) if is_anomaly else "normal"
        label_str    = f"{primary_type}_{object_name}"

        X_list.append(pixels)
        y_w_rel.append((label_str, is_anomaly))

    # Shuffle normal and anomalous samples independently within this category
    normal_idx  = [i for i, (_, r) in enumerate(y_w_rel) if not r]
    anomaly_idx = [i for i, (_, r) in enumerate(y_w_rel) if r]
    rng = np.random.default_rng(42)
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)
    order = normal_idx + anomaly_idx
    X_list  = [X_list[i]  for i in order]
    y_w_rel = [y_w_rel[i] for i in order]

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


def visa_setup_for_main(
    N_REL_CLASSES,                              # unused: relevance is determined by anomaly labels
    VERBOSE_FLAGS,
    seed,                                       # unused: no randomness needed during loading
    visa_root: str = "Datasets/VisA",
    object_categories: list[str] | None = None,
    img_size: tuple[int, int] = IMG_SIZE,
) -> tuple[np.ndarray, list[tuple[str, bool]], list[tuple[str, float]], list[str]]:
    """
    Load the full (multi-category) VisA dataset.

    Parameters
    ----------
    N_REL_CLASSES      : unused (relevance is determined directly from anomaly labels in CSV)
    VERBOSE_FLAGS      : list of flag ints; 0 in VERBOSE_FLAGS enables verbose output
    seed               : unused (no randomness in loading)
    visa_root          : path to VisA/ root directory
    object_categories  : list of object names to include; None → use all known categories
    img_size           : (width, height) resize target

    Returns
    -------
    X               : np.ndarray, shape (n_total_samples, flat_img_dim)
    y_w_rel         : list of (label_str, is_relevant_bool)
    sparsity_levels : list of (label, proportion) sorted by frequency (most → least common)
    relevant_labels : list of label strings that are marked as relevant (anomalous)
    """
    verbose = 0 in VERBOSE_FLAGS

    # ── npz cache ────────────────────────────────────────────────────────────
    # X saved as compressed .npz (float32 array); y_w_rel in a small sidecar pickle.
    cache_X   = os.path.join(visa_root, "visa_processed_X.npz")
    cache_y   = os.path.join(visa_root, "visa_processed_y.pkl")
    if os.path.exists(cache_X) and os.path.exists(cache_y):
        if verbose:
            print(f"Loading VisA dataset from cache: {cache_X} / {cache_y}")
        X = np.load(cache_X)["X"]
        with open(cache_y, "rb") as f:
            y_w_rel = pickle.load(f)
    else:
        if object_categories is None:
            object_categories = VISA_OBJECT_CATEGORIES

        all_X:      list[np.ndarray]          = []
        all_y_w_rel: list[tuple[str, bool]]   = []

        for obj in object_categories:
            if verbose:
                print(f"Loading VisA category: {obj} …")
            X_cat, y_cat = load_visa_category(visa_root, obj, img_size=img_size, verbose=verbose)
            all_X.append(X_cat)
            all_y_w_rel.extend(y_cat)

        X = np.concatenate(all_X, axis=0) if all_X else np.empty((0,))
        y_w_rel = all_y_w_rel

        if verbose:
            print(f"Saving VisA dataset to cache: {cache_X} / {cache_y}")
        np.savez_compressed(cache_X, X=X)
        with open(cache_y, "wb") as f:
            pickle.dump(y_w_rel, f)

    # ── shuffle within categories ──────────────────────────────────────────────
    X, y_w_rel = _shuffle_within_categories(X, y_w_rel, seed=42)

    # ── sparsity_levels ──────────────────────────────────────────────────────
    labels_only   = [lbl for lbl, _ in y_w_rel]
    label_counts  = Counter(labels_only)
    total         = len(labels_only)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in label_counts.most_common()]

    # ── relevant_labels ──────────────────────────────────────────────────────
    # A label is relevant if *any* sample with that label is marked anomalous.
    label_relevance: dict[str, bool] = {}
    for lbl, is_rel in y_w_rel:
        label_relevance[lbl] = label_relevance.get(lbl, False) or is_rel
    relevant_labels = sorted(lbl for lbl, rel in label_relevance.items() if rel)

    if verbose:
        print(f"\nVisA dataset loaded")
        print(f"  X shape        : {X.shape}")
        print(f"  Total samples  : {total}")
        print(f"  Total classes  : {len(label_counts)}")
        print(f"  Relevant labels ({len(relevant_labels)}): {relevant_labels[:10]} …")
        n_rel = sum(1 for _, r in y_w_rel if r)
        print(f"  Anomalous      : {n_rel} ({n_rel/total*100:.1f}%)")
        print(f"  Normal         : {total - n_rel} ({(total-n_rel)/total*100:.1f}%)")

    return X, y_w_rel, sparsity_levels, relevant_labels


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    VISA_ROOT = "Datasets/VisA"   # ← adjust to your local path

    X, y_w_rel, sparsity_levels, relevant_labels = visa_setup_for_main(
        N_REL_CLASSES=None,
        VERBOSE_FLAGS=[0],
        seed=42,
        visa_root=VISA_ROOT,
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