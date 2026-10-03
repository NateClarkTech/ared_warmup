#!/usr/bin/env python3
# VisA_DINO_AE_Data_Processing.py
#
# Pipeline: DINOv2 feature extraction (768-D) → simple autoencoder → 16-D latents
# Mirrors the parking lot DINOv2+AE pipeline exactly.
#
# Cache files written to visa_root/:
#   visa_dino_features.pkl   ← np.ndarray (N, 768), z-score normalized DINOv2 embeddings
#   visa_dino_ae_latents.pkl ← np.ndarray (N, 16),  autoencoder latents
#   visa_dino_ae_y.pkl       ← list of (label_str, is_relevant_bool)

import os
import numpy as np
import pandas as pd
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
# Config  (mirrors DINOv2_Parking_Lot.py)
# ──────────────────────────────────────────────────────────────────────────────

VISA_OBJECT_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

MODEL_NAME     = "facebook/dinov2-base"
DINO_IMG_SIZE  = (224, 224)
BATCH_SIZE_VIT = 32
BATCH_SIZE_AE  = 128

LATENT_DIM  = 16
ENC_HIDDEN  = [512, 256, 64]   # encoder hidden layers; decoder mirrors in reverse
EPOCHS      = 200
LR          = 1e-3
VAL_SPLIT   = 0.1
PATIENCE    = 15
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helper for DINOv2 extraction
# ──────────────────────────────────────────────────────────────────────────────

class _VisAImageDataset(Dataset):
    def __init__(self, img_paths, processor):
        self.img_paths = img_paths
        self.processor = processor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img    = Image.open(self.img_paths[idx]).convert("RGB").resize(
            DINO_IMG_SIZE, Image.LANCZOS
        )
        inputs = self.processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_dino_features(img_paths, device, verbose=False):
    """
    Pass all images through DINOv2 and return z-score-normalized CLS embeddings.
    Returns np.ndarray shape (N, 768), float32.
    """
    if verbose:
        print(f"Loading DINOv2 model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model_vit = AutoModel.from_pretrained(MODEL_NAME)
    model_vit.eval()
    model_vit.to(device)

    dataset    = _VisAImageDataset(img_paths, processor)
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

    # z-score normalization per dimension — same as parking lot pipeline
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    return X


# ──────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

class _SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_hidden, device):
        super().__init__()
        self._device = device

        # Encoder: input_dim → enc_hidden[0] → ... → latent_dim
        enc_layers = []
        prev = input_dim
        for h in enc_hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent_dim → enc_hidden[-1] → ... → input_dim
        dec_layers = []
        prev = latent_dim
        for h in reversed(enc_hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.to(device)

    def forward(self, x):
        x     = x.to(self._device)
        z     = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


def _train_autoencoder(X, device, verbose=False):
    """
    Train the autoencoder on X (N, 768) and return 16-D latents (N, 16).
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    N         = X.shape[0]
    indices   = np.random.RandomState(RANDOM_SEED).permutation(N)
    train_idx = indices[int(N * VAL_SPLIT):]
    val_idx   = indices[:int(N * VAL_SPLIT)]

    X_train = torch.from_numpy(X[train_idx]).float()
    X_val   = torch.from_numpy(X[val_idx]).float()

    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE_AE, shuffle=True)
    val_loader   = DataLoader(X_val,   batch_size=BATCH_SIZE_AE, shuffle=False)

    model     = _SimpleAutoencoder(X.shape[1], LATENT_DIM, ENC_HIDDEN, device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    if verbose:
        print(f"\nTraining autoencoder  768 -> {LATENT_DIM}  "
              f"(hidden: {ENC_HIDDEN}, max epochs: {EPOCHS}, patience: {PATIENCE})")

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, x_rec = model(batch)
            loss = criterion(x_rec, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, x_rec = model(batch)
                val_loss += criterion(x_rec, batch).item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        if verbose and (epoch <= 10 or epoch % 10 == 0):
            print(f"  Epoch {epoch:4d}/{EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    if verbose:
        print("Autoencoder training complete.")

    # extract latents for all N samples
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        z_all, _ = model(X_tensor)
        z_all    = z_all.cpu().numpy()

    return z_all.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# VisA annotation parser
# ──────────────────────────────────────────────────────────────────────────────

def _collect_visa_category(visa_root, object_name):
    anno_path = os.path.join(visa_root, object_name, "image_anno.csv")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(
            f"Annotation CSV not found: {anno_path}\n"
            f"Expected VisA structure: {visa_root}/<object>/image_anno.csv"
        )

    df        = pd.read_csv(anno_path)
    img_paths = []
    y_w_rel   = []

    for _, row in df.iterrows():
        abs_path  = os.path.join(visa_root, row["image"])
        raw_label = str(row["label"])

        if not os.path.exists(abs_path):
            print(f"  [WARN] Image not found, skipping: {abs_path}")
            continue

        is_anomaly   = raw_label.lower() != "normal"
        primary_type = raw_label.split(",")[0].strip() if is_anomaly else "normal"
        label_str    = f"{primary_type}_{object_name}"

        img_paths.append(abs_path)
        y_w_rel.append((label_str, is_anomaly))

    return img_paths, y_w_rel


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def visa_dino_ae_setup_for_main(
    N_REL_CLASSES,                              # unused: relevance determined by anomaly labels
    VERBOSE_FLAGS,
    seed,                                       # unused: fixed seeds inside pipeline
    visa_root="Datasets/VisA",
    object_categories=None,
):
    """
    Load the VisA dataset as 16-D autoencoder latents derived from DINOv2 embeddings.

    Pipeline (run once, then cached):
      1. Collect image paths + labels from each object's image_anno.csv
      2. Extract DINOv2 CLS token embeddings  -> (N, 768), z-score normalized
         [cached to visa_dino_features.pkl]
      3. Train simple autoencoder 768->16      -> (N, 16) latents
         [cached to visa_dino_ae_latents.pkl + visa_dino_ae_y.pkl]

    Parameters
    ----------
    N_REL_CLASSES      : unused
    VERBOSE_FLAGS      : list of flag ints; 0 in VERBOSE_FLAGS enables verbose output
    seed               : unused (fixed seeds used internally)
    visa_root          : path to VisA/ root directory
    object_categories  : object names to include; None uses all known categories

    Returns
    -------
    X               : np.ndarray, shape (N, 16)
    y_w_rel         : list of (label_str, is_relevant_bool)
    sparsity_levels : list of (label, proportion) sorted most -> least common
    relevant_labels : list of anomalous label strings
    """
    verbose = 0 in VERBOSE_FLAGS
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    cache_feat    = os.path.join(visa_root, "visa_dino_features.pkl")
    cache_latents = os.path.join(visa_root, "visa_dino_ae_latents.pkl")
    cache_y       = os.path.join(visa_root, "visa_dino_ae_y.pkl")

    # ── load fully cached result if available ─────────────────────────────────
    if os.path.exists(cache_feat) and os.path.exists(cache_y):
        if verbose:
            print(f"Loading VisA-DINO-AE latents from cache:\n"
                  f"  {cache_feat}\n  {cache_y}")
        with open(cache_feat, "rb") as f:
            X = pickle.load(f)
        with open(cache_y, "rb") as f:
            y_w_rel = pickle.load(f)

    else:
        if object_categories is None:
            object_categories = VISA_OBJECT_CATEGORIES

        # collect image paths + labels
        all_img_paths = []
        all_y_w_rel   = []
        for obj in object_categories:
            if verbose:
                print(f"Collecting VisA category: {obj} ...")
            paths, y_cat = _collect_visa_category(visa_root, obj)
            all_img_paths.extend(paths)
            all_y_w_rel.extend(y_cat)

        if verbose:
            print(f"\nTotal images: {len(all_img_paths)}")

        # DINOv2 extraction — reuse cache if the features step already ran
        if os.path.exists(cache_feat):
            if verbose:
                print(f"Loading cached DINOv2 features: {cache_feat}")
            with open(cache_feat, "rb") as f:
                X_dino = pickle.load(f)
        else:
            if verbose:
                print("Extracting DINOv2 features ...")
            X_dino = _extract_dino_features(all_img_paths, device, verbose=verbose)
            if verbose:
                print(f"Saving DINOv2 features to cache: {cache_feat}")
            with open(cache_feat, "wb") as f:
                pickle.dump(X_dino, f)

        if verbose:
            print(f"DINOv2 features shape: {X_dino.shape}")

        # autoencoder -> 16-D latents
        X       = X_dino # _train_autoencoder(X_dino, device, verbose=verbose)
        y_w_rel = all_y_w_rel

        if verbose:
            print(f"Saving latents to cache:\n  {cache_latents}\n  {cache_y}")
        with open(cache_latents, "wb") as f:
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
        print(f"\nVisA-DINO dataset ready")
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
    VISA_ROOT = "Datasets/VisA"   # <- adjust to your local path

    X, y_w_rel, sparsity_levels, relevant_labels = visa_dino_ae_setup_for_main(
        N_REL_CLASSES=None,
        VERBOSE_FLAGS=[0],
        seed=42,
        visa_root=VISA_ROOT,
    )

    total     = len(y_w_rel)
    n_anomaly = sum(1 for _, r in y_w_rel if r)

    print("\n-- Summary --------------------------------------------------")
    print(f"X shape        : {X.shape}")
    print(f"Total samples  : {total}")
    print(f"Anomaly rate   : {n_anomaly/total*100:.2f}%  ({n_anomaly} anomalous / {total - n_anomaly} normal)")
    print()
    for lbl, prop in sparsity_levels:
        print(f"  {lbl:<45} {prop*100:.2f}%")