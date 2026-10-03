import pickle
import numpy as np
from collections import Counter

def parking_lot_dagmm_preprocessed(VERBOSE_FLAGS=None, seed=42):
    """
    Load the DAGMM-preprocessed Parking Lot data.
    Relevance is determined directly from the loaded y_w_rel (no hard-coding).

    Returns
    -------
    X                : np.ndarray, shape (n_samples, latent_dim)
    y_w_rel          : list of tuples (label_str, is_relevant_bool)
    sparsity_levels  : list of (label, proportion) sorted by frequency
    relevant_labels  : list of str - actually relevant class names
    """
    if VERBOSE_FLAGS is None:
        VERBOSE_FLAGS = []

    # Paths – adjust if needed
    # latent_path = "Parking_Lot_Data/preprocessed_X_latent_NREL8.pkl"
    y_path      = "Datasets/Parking_Lot_Data/y_w_rel_NREL8.pkl"
    #
    # # Load data
    # with open(latent_path, "rb") as f:
    #     X = pickle.load(f)

    with open("results/preprocessed_X_full_z_NREL8.pkl", "rb") as f:
        X = pickle.load(f)  # ← this is now z_all (66-dim)

    with open(y_path, "rb") as f:
        y_w_rel = pickle.load(f)  # Keep original relevance flags!

    X = np.array(X, dtype=np.float32)

    # Extract labels and relevance
    labels_only = [lbl for lbl, _ in y_w_rel]
    relevance_flags = [is_rel for _, is_rel in y_w_rel]

    # Determine which labels are actually marked as relevant
    label_to_relevance = dict(y_w_rel)  # assumes no duplicates, which should be fine
    relevant_labels = sorted([lbl for lbl, is_rel in label_to_relevance.items() if is_rel])

    # Count frequencies
    label_counts = Counter(labels_only)
    total = len(labels_only)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in label_counts.most_common()]

    # Verbose output
    if 0 in VERBOSE_FLAGS:
        print(f"Loaded DAGMM-preprocessed Parking Lot data")
        print(f"   X shape         : {X.shape}")
        print(f"   n_samples       : {len(y_w_rel)}")
        print(f"   latent dim      : {X.shape[1]}")
        print(f"   Total classes   : {len(label_counts)}")
        print(f"   Relevant classes ({len(relevant_labels)}): {relevant_labels}")
        print(f"   Relevance distribution:")
        rel_count = sum(1 for _, r in y_w_rel if r)
        print(f"     Relevant points   : {rel_count} ({rel_count/total*100:.2f}%)")
        print(f"     Irrelevant points : {total - rel_count} ({100 - rel_count/total*100:.2f}%)")

    return X, y_w_rel, sparsity_levels, relevant_labels