import numpy as np
import pickle
import pandas as pd
import cv2
from collections import Counter
import os


def parking_lot_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed):
    """
    Load the raw Parking Lot dataset and return everything in the exact format
    expected by A/RED, including dynamically choosing the N_REL_CLASSES rarest
    classes as "relevant".

    This now mirrors the behavior of MNIST_setup_for_main().

    Parameters
    ----------
    N_REL_CLASSES : int
        Number of rarest classes to mark as "relevant" (anomalies of interest).
    VERBOSE_FLAGS : list of int
        Used for optional printing (0 = basic info).
    seed : int
        Random seed (currently not used for sampling, but kept for API consistency).

    Returns
    -------
    X                : np.ndarray, (n_samples, 16384)  [128×128 flattened grayscale]
    y_w_rel          : list of (label_str, is_relevant_bool)
    sparsity_levels  : list of (label, proportion) sorted from most → least common
    relevant_labels  : list of str  (the N rarest classes)
    """
    # === Load Dataset ===
    features_path = "Datasets/Parking_Lot_Data/features.pkl"
    labels_path   = "Datasets/Parking_Lot_Data/labels.csv"

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Parking lot data not found. Check ./Parking_Lot_Data/")

    with open(features_path, 'rb') as f:
        features = pickle.load(f)                     # (4410, 128, 128, 3)

    labels_df = pd.read_csv(labels_path)
    y = labels_df["label"].tolist()

    # Convert to grayscale, normalize to [0,1], and flatten
    print("Converting images to grayscale and flattening...")
    X = np.array([
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32).flatten() / 255.0
        for img in features
    ])

    assert len(X) == len(y), "Feature-label length mismatch"

    # === Count occurrences of each class ===
    label_counts = Counter(y)

    # === Sort classes by frequency: most common → least common ===
    sorted_counts = label_counts.most_common()                     # highest first
    sorted_labels = [lbl for lbl, cnt in sorted_counts]

    # === Select the N_REL_CLASSES rarest classes as relevant ===
    relevant_labels = sorted_labels[-N_REL_CLASSES:]               # last N = rarest
    relevant_set = set(relevant_labels)

    # === Build y_w_rel ===
    y_w_rel = [(label, label in relevant_set) for label in y]

    # === Sparsity levels (proportion of each class, most → least common) ===
    total = len(y)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in sorted_counts]

    # === Verbose output ===
    if 0 in VERBOSE_FLAGS:
        print(f"\nRunning on Parking Lot dataset with {len(y)} events")
        print(f"Total classes found: {len(label_counts)}")
        print(f"Class distribution (top 10 most common):")
        for lbl, cnt in sorted_counts[:10]:
            print(f"   {lbl:20s} : {cnt:4d} ({cnt/total*100:5.2f}%)")
        print(f"\n→ Selected {N_REL_CLASSES} rarest classes as RELEVANT:")
        for lbl in relevant_labels:
            cnt = label_counts[lbl]
            print(f"   • {lbl:20s} : {cnt:4d} samples")
        print()

    return X, y_w_rel, sparsity_levels, relevant_labels