import os
import numpy as np
import pandas as pd
from collections import Counter
from DINOv2_Parking_Lot import run_dinov2_autoencoder_preprocessing  # adjust name if needed

def parking_lot_dino_preprocessed(N_REL_CLASSES, VERBOSE_FLAGS=None, seed=42):
    if VERBOSE_FLAGS is None:
        VERBOSE_FLAGS = []

    features_path = "Datasets/Parking_Lot_Data/PARKING_LOT_DINO_latents_16d.npy"
    labels_path   = "Datasets/Parking_Lot_Data/labels.csv"

    # If files don't exist, generate them via DINOv2+AE
    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        print(
            "DINOv2 preprocessed files not found. "
            "Running DINOv2+Autoencoder preprocessing to generate them..."
        )
        run_dinov2_autoencoder_preprocessing(out_dir="Datasets/Parking_Lot_Data", do_viz=False)

    # Re-check after running the preprocessor
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Feature file still not found at {features_path} even after preprocessing."
        )
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels CSV still not found at {labels_path} even after preprocessing."
        )

    # X_skewed: (n_samples, 16) — latent features
    X_skewed = np.load(features_path).astype(np.float32)

    labels_df = pd.read_csv(labels_path)
    y = labels_df["label"].astype(str).to_numpy()

    assert X_skewed.shape[0] == len(y), (
        f"Feature-label length mismatch: X has {X_skewed.shape[0]} rows, "
        f"labels have {len(y)} entries."
    )

    # --- Define relevance as N_REL_CLASSES rarest labels ---
    label_counts = Counter(y)
    sorted_counts = label_counts.most_common()  # most → least frequent
    sorted_labels = [lbl for lbl, _ in sorted_counts]

    rel_classes = sorted_labels[-N_REL_CLASSES:]
    relevant_set = set(rel_classes)

    relevance = np.array([lbl in relevant_set for lbl in y], dtype=bool)
    y_w_rel = np.empty((len(y), 2), dtype=object)
    y_w_rel[:, 0] = y
    y_w_rel[:, 1] = relevance

    total = len(y)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in sorted_counts]

    if 0 in VERBOSE_FLAGS:
        print(f"\nRunning on Parking Lot DINO 16D preprocessed latent dataset with {len(y)} events")
        print(f"Total classes found: {len(label_counts)}")
        print("Class distribution (top 10 most common):")
        for lbl, cnt in sorted_counts[:10]:
            print(f"   {lbl:20s} : {cnt:4d} ({cnt/total*100:5.2f}%)")
        print(f"\n→ Selected {N_REL_CLASSES} rarest classes as RELEVANT:")
        for lbl in rel_classes:
            cnt = label_counts.get(lbl, 0)
            print(f"   • {lbl:20s} : {cnt:4d} samples")
        print()

    return X_skewed, y_w_rel, sparsity_levels, rel_classes
