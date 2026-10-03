#!/usr/bin/env python3
# DINOv2_MVtechAD_Processing.py
#
# Cache files written to mvtechad_root/:
#   mvtechad_dino_features.pkl   <- np.ndarray (N, 768), z-score normalized DINOv2 embeddings
#   mvtechad_dino_ae_latents.pkl <- np.ndarray (N, 16),  autoencoder latents
#   mvtechad_dino_ae_y.pkl       <- list of (label_str, is_relevant_bool)
#
# label_str   = "<defect_type>_<object>"  (anomalous, defect_type = folder name)
#             = "normal_<object>"          (normal, from train/good or test/good)
# is_relevant = True for anomalous samples, False for normal
#
# Data ordering — controlled by INTERLEAVED_CATEGORIES flag:
#
#   INTERLEAVED_CATEGORIES = False  (new default):
#     - Training samples (train/good) kept ordered by category
#     - Test samples (all test/<label>/ folders) globally shuffled across categories
#     - Mirrors realistic deployment: train on normals first, then encounter defects
#
#   INTERLEAVED_CATEGORIES = True  (old behaviour):
#     - All samples collected category by category (train/good then test/*)
#     - No global shuffle; categories are fully interleaved in order

import os
import numpy as np
import pickle
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MVTECHAD_OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MODEL_NAME     = "facebook/dinov2-base"
DINO_IMG_SIZE  = (224, 224)
BATCH_SIZE_VIT = 32

# ──────────────────────────────────────────────────────────────────────────────
# Data ordering flag
#
#   False -> new ordering: ordered train block + globally shuffled test block
#   True  -> old ordering: category-by-category (train/good then test/*), no global shuffle
# ──────────────────────────────────────────────────────────────────────────────
INTERLEAVED_CATEGORIES = True


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helper
# ──────────────────────────────────────────────────────────────────────────────

class _MVtechImageDataset(Dataset):
    def __init__(self, img_paths, processor):
        self.img_paths = img_paths
        self.processor = processor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB").resize(
            DINO_IMG_SIZE, Image.LANCZOS
        )
        inputs = self.processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_dino_features(img_paths, device, verbose=False):
    """Extract z-score normalized DINOv2 CLS embeddings."""
    if verbose:
        print(f"Loading DINOv2 model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model_vit = AutoModel.from_pretrained(MODEL_NAME)
    model_vit.eval()
    model_vit.to(device)

    dataset = _MVtechImageDataset(img_paths, processor)
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
            outputs = model_vit(pixel_values)
            cls_tokens = outputs.last_hidden_state[:, 0, :]   # (B, 768)
            features_list.append(cls_tokens.cpu().numpy())
            if verbose and (i % 10 == 0 or i == total):
                print(f"  DINOv2 batch {i}/{total}")

    X = np.vstack(features_list).astype(np.float32)

    # z-score normalization per dimension
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    return X


# ──────────────────────────────────────────────────────────────────────────────
# Image path + label collector (used by old/interleaved ordering only)
# ──────────────────────────────────────────────────────────────────────────────

def _collect_mvtechad_category(mvtechad_root, object_name, include_train=True):
    """Walk one MVtechAD object category and return image paths + labels."""
    obj_dir = os.path.join(mvtechad_root, object_name)
    if not os.path.isdir(obj_dir):
        raise FileNotFoundError(
            f"MVtechAD object directory not found: {obj_dir}\n"
            f"Expected structure: {mvtechad_root}/<object>/test/<label>/"
        )

    img_paths = []
    y_w_rel = []

    def _ingest(folder, label_str, is_anomaly):
        if not os.path.isdir(folder):
            return
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            img_paths.append(os.path.join(folder, fname))
            y_w_rel.append((label_str, is_anomaly))

    if include_train:
        _ingest(os.path.join(obj_dir, "train", "good"),
                f"normal_{object_name}", is_anomaly=False)

    test_dir = os.path.join(obj_dir, "test")
    if os.path.isdir(test_dir):
        for label_folder in sorted(os.listdir(test_dir)):
            label_path = os.path.join(test_dir, label_folder)
            if not os.path.isdir(label_path):
                continue
            if label_folder.lower() == "good":
                _ingest(label_path, f"normal_{object_name}", is_anomaly=False)
            else:
                _ingest(label_path, f"{label_folder}_{object_name}", is_anomaly=True)

    return img_paths, y_w_rel


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def mvtechad_dino_ae_setup_for_main(
    N_REL_CLASSES,                              # unused: relevance determined by anomaly labels
    VERBOSE_FLAGS,
    seed,
    mvtechad_root: str = None,
    object_categories=None,
    include_train=True,
):
    """
    Load the MVtechAD dataset as DINOv2 embeddings (768-D, z-score normalised).

    Data ordering is controlled by the module-level INTERLEAVED_CATEGORIES flag:

      INTERLEAVED_CATEGORIES = False  (new default, recommended):
        Training samples (train/good/) kept in category order so ARED sees all
        normals for each category before moving on. Test samples from ALL
        categories are pooled and globally shuffled before being appended.

      INTERLEAVED_CATEGORIES = True  (old behaviour):
        All samples collected category by category (train/good then test/*) with
        no global shuffle — original behaviour from the first paper.

    The cache filename includes the ordering mode so switching the flag
    automatically regenerates the cache rather than loading stale data.

    Parameters
    ----------
    N_REL_CLASSES      : unused
    VERBOSE_FLAGS      : list of flag ints; 0 in VERBOSE_FLAGS enables verbose output
    seed               : RNG seed for test-pool shuffle (new ordering only)
    mvtechad_root      : path to MVtechAD/ root; None triggers auto-detection
    object_categories  : object names to include; None uses all known categories
    include_train      : if True, also include train/good/ as normal samples

    Returns
    -------
    X               : np.ndarray, shape (N, 768)
    y_w_rel         : list of (label_str, is_relevant_bool)
    sparsity_levels : list of (label, proportion) sorted most -> least common
    relevant_labels : list of anomalous label strings
    """
    verbose = 0 in VERBOSE_FLAGS

    # ── Smart Dataset Path Detection ──────────────────────────────────────────
    if mvtechad_root is None:
        nate_path  = "Datasets/MVtechAD"
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
                "  mvtechad_dino_ae_setup_for_main(mvtechad_root='/your/path')"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")
        mode_str = "interleaved (old)" if INTERLEAVED_CATEGORIES else "train-then-shuffled-test (new)"
        print(f"Data ordering mode: {mode_str}  [INTERLEAVED_CATEGORIES={INTERLEAVED_CATEGORIES}]")

    # Cache filenames encode the ordering mode so flipping the flag forces a rebuild
    mode_tag   = "interleaved" if INTERLEAVED_CATEGORIES else "split_shuffle"
    cache_feat = os.path.join(mvtechad_root, f"mvtechad_dino_features_{mode_tag}.pkl")
    cache_y    = os.path.join(mvtechad_root, f"mvtechad_dino_y_{mode_tag}.pkl")

    # ── Load from cache if available ──────────────────────────────────────────
    if os.path.exists(cache_feat) and os.path.exists(cache_y):
        if verbose:
            print(f"Loading cached MVtechAD DINOv2 data from:\n  {cache_feat}")
        with open(cache_feat, "rb") as f:
            X = pickle.load(f)
        with open(cache_y, "rb") as f:
            y_w_rel = pickle.load(f)

    else:
        if object_categories is None:
            object_categories = MVTECHAD_OBJECT_CATEGORIES

        if INTERLEAVED_CATEGORIES:
            # ── OLD ordering: category-by-category, no global shuffle ─────────
            if verbose:
                print("Collecting data in interleaved category order (old behaviour) ...")
            all_img_paths = []
            all_y_w_rel   = []
            for obj in object_categories:
                if verbose:
                    print(f"  Collecting category: {obj} ...")
                paths, y_cat = _collect_mvtechad_category(
                    mvtechad_root, obj, include_train=include_train
                )
                all_img_paths.extend(paths)
                all_y_w_rel.extend(y_cat)
            if verbose:
                print(f"Total images: {len(all_img_paths)}")

        else:
            # ── NEW ordering: ordered train block + globally shuffled test ────
            if verbose:
                print("Collecting data with ordered train block + shuffled test block (new behaviour) ...")
            train_img_paths, train_y = [], []
            test_img_paths,  test_y  = [], []

            for obj in object_categories:
                if verbose:
                    print(f"  Collecting category: {obj} ...")

                obj_dir = os.path.join(mvtechad_root, obj)
                if not os.path.isdir(obj_dir):
                    seen_root = os.path.abspath(mvtechad_root) if os.path.isdir(mvtechad_root) else None
                    if seen_root:
                        seen_entries = sorted(os.listdir(seen_root))
                        seen_str = "\n  ".join(seen_entries) if seen_entries else "(empty)"
                    else:
                        cwd_entries = sorted(os.listdir("."))
                        seen_str    = "\n  ".join(cwd_entries) if cwd_entries else "(empty)"
                        seen_root   = (f"{os.path.abspath('.')}  "
                                       f"(mvtechad_root '{mvtechad_root}' not found, showing cwd)")
                    raise FileNotFoundError(
                        f"Object directory not found: {os.path.abspath(obj_dir)}\n"
                        f"Expected MVtechAD structure: {mvtechad_root}/<object>/test/<label>/\n"
                        f"Contents of {seen_root}:\n  {seen_str}"
                    )

                # train/good -> ordered train block
                if include_train:
                    train_folder = os.path.join(obj_dir, "train", "good")
                    if os.path.isdir(train_folder):
                        for fname in sorted(os.listdir(train_folder)):
                            if not fname.lower().endswith(
                                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                            ):
                                continue
                            train_img_paths.append(os.path.join(train_folder, fname))
                            train_y.append((f"normal_{obj}", False))

                # test/<label>/ -> test pool (shuffled below)
                test_dir = os.path.join(obj_dir, "test")
                if not os.path.isdir(test_dir):
                    print(f"  [WARN] No test/ directory found for {obj}, skipping.")
                    continue
                for label_folder in sorted(os.listdir(test_dir)):
                    label_path = os.path.join(test_dir, label_folder)
                    if not os.path.isdir(label_path):
                        continue
                    is_normal = label_folder.lower() == "good"
                    label_str = (f"normal_{obj}" if is_normal
                                 else f"{label_folder}_{obj}")
                    for fname in sorted(os.listdir(label_path)):
                        if not fname.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                        ):
                            continue
                        test_img_paths.append(os.path.join(label_path, fname))
                        test_y.append((label_str, not is_normal))

            # globally shuffle the test pool
            rng          = np.random.RandomState(seed)
            test_indices = rng.permutation(len(test_img_paths))
            test_img_paths = [test_img_paths[i] for i in test_indices]
            test_y         = [test_y[i]         for i in test_indices]

            if verbose:
                print(f"\nTrain block : {len(train_img_paths)} images (ordered by category)")
                print(f"Test block  : {len(test_img_paths)} images (globally shuffled)")

            all_img_paths = train_img_paths + test_img_paths
            all_y_w_rel   = train_y         + test_y

            if verbose:
                print(f"Total images: {len(all_img_paths)}")

        # ── DINOv2 extraction ─────────────────────────────────────────────────
        if verbose:
            print("Extracting DINOv2 features ...")
        X = _extract_dino_features(all_img_paths, device, verbose=verbose)

        if verbose:
            print(f"DINOv2 features shape: {X.shape}")
            print(f"Saving cache to:\n  {cache_feat}\n  {cache_y}")

        y_w_rel = all_y_w_rel

        with open(cache_feat, "wb") as f:
            pickle.dump(X, f)
        with open(cache_y, "wb") as f:
            pickle.dump(y_w_rel, f)

    # ── Sparsity levels ───────────────────────────────────────────────────────
    labels_only  = [lbl for lbl, _ in y_w_rel]
    label_counts = Counter(labels_only)
    total        = len(labels_only)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in label_counts.most_common()]

    # ── Relevant labels ───────────────────────────────────────────────────────
    label_relevance = {}
    for lbl, is_rel in y_w_rel:
        label_relevance[lbl] = label_relevance.get(lbl, False) or is_rel
    relevant_labels = sorted(lbl for lbl, rel in label_relevance.items() if rel)

    if verbose:
        print(f"\nMVtechAD-DINO dataset ready")
        print(f"  X shape        : {X.shape}")
        print(f"  Total samples  : {total}")
        print(f"  Total classes  : {len(label_counts)}")
        n_rel = sum(1 for _, r in y_w_rel if r)
        print(f"  Anomalous      : {n_rel} ({n_rel/total*100:.1f}%)")
        print(f"  Normal         : {total - n_rel} ({(total-n_rel)/total*100:.1f}%)")
        print(f"  Relevant labels ({len(relevant_labels)}): {relevant_labels[:10]} ...")

    return X, y_w_rel, sparsity_levels, relevant_labels


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y_w_rel, sparsity_levels, relevant_labels = mvtechad_dino_ae_setup_for_main(
        N_REL_CLASSES=None,
        VERBOSE_FLAGS=[0],
        seed=42,
    )

    total     = len(y_w_rel)
    n_anomaly = sum(1 for _, r in y_w_rel if r)

    print("\n-- Summary --------------------------------------------------")
    print(f"X shape        : {X.shape}")
    print(f"Total samples  : {total}")
    print(f"Anomaly rate   : {n_anomaly/total*100:.2f}%")
    print()
    for lbl, prop in sparsity_levels[:10]:
        print(f"  {lbl:<45} {prop*100:.2f}%")