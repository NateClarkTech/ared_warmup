#!/usr/bin/env python3
# EMNIST_DINO_Data_Processing.py
#
# Pipeline: EMNIST → DINOv2 feature extraction (768-D CLS token embeddings)
#
# EMNIST images are 28×28 grayscale; they are upscaled to 224×224 and converted
# to 3-channel RGB before passing through the DINOv2 processor.
#
# Cache file (written alongside the emnist.pkl source file):
#   emnist_dino_features.pkl ← np.ndarray (N, 768), z-score normalized DINOv2 embeddings
#   emnist_dino_y.pkl        ← list of (label_str, is_relevant_bool)

import os
import numpy as np
import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

EMNIST_LABEL_TO_ASCII = [
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122
]

MODEL_NAME     = "facebook/dinov2-base"
DINO_IMG_SIZE  = (224, 224)
BATCH_SIZE_VIT = 32


# ──────────────────────────────────────────────────────────────────────────────
# Label helpers
# ──────────────────────────────────────────────────────────────────────────────

def _label_to_char(label):
    label_int = int(label)
    if 0 <= label_int < len(EMNIST_LABEL_TO_ASCII):
        return chr(EMNIST_LABEL_TO_ASCII[label_int])
    raise ValueError(f"Invalid EMNIST label index: {label_int}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helper — converts flat 28×28 grayscale arrays to 224×224 RGB tensors
# ──────────────────────────────────────────────────────────────────────────────

class _EMNISTImageDataset(Dataset):
    """
    Wraps the raw EMNIST image array (N, 784) for DINOv2 feature extraction.
    Each 28×28 grayscale image is:
      1. Reshaped to (28, 28)
      2. Upscaled to 224×224 with LANCZOS
      3. Converted to RGB (3-channel) so DINOv2 processor accepts it
    """

    def __init__(self, X_flat, processor):
        self.X_flat    = X_flat
        self.processor = processor

    def __len__(self):
        return len(self.X_flat)

    def __getitem__(self, idx):
        arr = self.X_flat[idx]

        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        img = Image.fromarray(arr.reshape(28, 28), mode="L")
        img = img.resize(DINO_IMG_SIZE, Image.LANCZOS).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_dino_features(X_flat, device, verbose=False):
    """
    Pass all EMNIST images through DINOv2 and return z-score-normalized
    CLS token embeddings.

    Returns np.ndarray shape (N, 768), float32.
    """
    if verbose:
        print(f"Loading DINOv2 model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model_vit = AutoModel.from_pretrained(MODEL_NAME)
    model_vit.eval()
    model_vit.to(device)

    dataset    = _EMNISTImageDataset(X_flat, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_VIT,
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    features_list = []
    total = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            pixel_values = batch["pixel_values"].to(device)
            outputs      = model_vit(pixel_values)
            cls_tokens   = outputs.last_hidden_state[:, 0, :]   # (B, 768)
            features_list.append(cls_tokens.cpu().numpy())
            if verbose and (i % 10 == 0 or i == total):
                print(f"  DINOv2 batch {i}/{total}")

    X = np.vstack(features_list).astype(np.float32)

    # z-score normalization per dimension
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    return X


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def EMNIST_dino_setup_for_main(
    N_REL_CLASSES,
    VERBOSE_FLAGS,
    seed=42,                        # unused; kept for API consistency
    save_path="Datasets/EMNIST/emnist.pkl",
):
    """
    Load EMNIST and return 768-D DINOv2 CLS token embeddings with the same
    4-tuple interface as all other dataset loaders.

    Pipeline (run once, then cached):
      1. Load raw EMNIST images from emnist.pkl
      2. Upscale 28×28 grayscale → 224×224 RGB
      3. Extract DINOv2 CLS embeddings → (N, 768), z-score normalized
         [cached: emnist_dino_features.pkl + emnist_dino_y.pkl]

    Relevance: the N_REL_CLASSES least-common character classes are marked
    relevant, matching the original EMNIST_setup_for_main behaviour.

    Parameters
    ----------
    N_REL_CLASSES : number of rarest classes to mark as relevant
    VERBOSE_FLAGS : list of flag ints; 0 enables verbose output
    seed          : unused (kept for API consistency)
    save_path     : path to emnist.pkl

    Returns
    -------
    X               : np.ndarray, shape (N, 768)
    y_w_rel         : list of (label_str, is_relevant_bool)
    sparsity_levels : list of (label, proportion) sorted most -> least common
    relevant_labels : list of label strings marked as relevant
    """
    verbose   = 0 in VERBOSE_FLAGS
    cache_dir = os.path.dirname(os.path.abspath(save_path))

    cache_feat = os.path.join(cache_dir, "emnist_dino_features.pkl")
    cache_y    = os.path.join(cache_dir, "emnist_dino_y.pkl")

    # ── load from cache if available ──────────────────────────────────────────
    if os.path.exists(cache_feat) and os.path.exists(cache_y):
        if verbose:
            print(f"Loading EMNIST-DINO features from cache:\n"
                  f"  {cache_feat}\n  {cache_y}")
        with open(cache_feat, "rb") as f:
            X = pickle.load(f)
        with open(cache_y, "rb") as f:
            y_w_rel_cached = pickle.load(f)  # only used to recover label strings

        # Recompute relevance from scratch so N_REL_CLASSES is always respected
        y_chars = [lbl for lbl, _ in y_w_rel_cached]
        class_counts = Counter(y_chars)
        least_common = [cls for cls, _ in class_counts.most_common()[-N_REL_CLASSES:]]
        relevant_set = set(least_common)
        y_w_rel = [(ch, ch in relevant_set) for ch in y_chars]

    else:
        # -- Load raw EMNIST --------------------------------------------------
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"EMNIST file not found at {save_path}. "
                "Please generate it first using the EMNIST loading script."
            )
        if verbose:
            print(f"Loading EMNIST from {save_path} ...")
        with open(save_path, "rb") as f:
            data = pickle.load(f)

        X_raw  = data["images"]   # (N, 784)
        y_raw  = data["labels"]   # (N,)
        y_chars = [_label_to_char(lbl) for lbl in y_raw]

        if verbose:
            print(f"EMNIST loaded: {len(y_chars)} samples")

        # -- Determine relevance (N rarest classes) ---------------------------
        class_counts = Counter(y_chars)
        least_common = [cls for cls, _ in class_counts.most_common()[-N_REL_CLASSES:]]
        relevant_set = set(least_common)
        y_w_rel      = [(ch, ch in relevant_set) for ch in y_chars]

        # -- DINOv2 extraction ------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Using device: {device}")
            print("Extracting DINOv2 features ...")

        X = _extract_dino_features(X_raw, device, verbose=verbose)

        if verbose:
            print(f"DINOv2 features shape: {X.shape}")
            print(f"Saving to cache:\n  {cache_feat}\n  {cache_y}")

        with open(cache_feat, "wb") as f:
            pickle.dump(X, f)
        with open(cache_y, "wb") as f:
            pickle.dump(y_w_rel, f)

    # ── sparsity_levels ───────────────────────────────────────────────────────
    labels_only  = [lbl for lbl, _ in y_w_rel]
    label_counts = Counter(labels_only)
    total        = len(labels_only)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in label_counts.most_common()]

    # ── relevant_labels ───────────────────────────────────────────────────────
    label_relevance = {}
    for lbl, is_rel in y_w_rel:
        label_relevance[lbl] = label_relevance.get(lbl, False) or is_rel
    relevant_labels = sorted(lbl for lbl, rel in label_relevance.items() if rel)

    if verbose:
        print(f"\nEMNIST-DINO dataset ready")
        print(f"  X shape        : {X.shape}")
        print(f"  Total samples  : {total}")
        print(f"  Total classes  : {len(label_counts)}")
        n_rel = sum(1 for _, r in y_w_rel if r)
        print(f"  Relevant       : {n_rel} ({n_rel/total*100:.1f}%)")
        print(f"  Normal         : {total - n_rel} ({(total-n_rel)/total*100:.1f}%)")
        print(f"  Relevant labels: {relevant_labels}")

    return X, y_w_rel, sparsity_levels, relevant_labels


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_REL_CLASSES = 2
    VERBOSE_FLAGS = [0]

    X, y_w_rel, sparsity_levels, relevant_labels = EMNIST_dino_setup_for_main(
        N_REL_CLASSES=N_REL_CLASSES,
        VERBOSE_FLAGS=VERBOSE_FLAGS,
        save_path="Datasets/EMNIST/emnist.pkl",
    )

    total     = len(y_w_rel)
    n_anomaly = sum(1 for _, r in y_w_rel if r)

    print("\n-- Summary --------------------------------------------------")
    print(f"X shape        : {X.shape}")
    print(f"Total samples  : {total}")
    print(f"Relevant labels: {relevant_labels}")
    print(f"Anomaly rate   : {n_anomaly/total*100:.2f}%  ({n_anomaly} relevant / {total - n_anomaly} normal)")
    print()
    for lbl, prop in sparsity_levels[:15]:
        print(f"  {lbl:<10} {prop*100:.2f}%")