#!/usr/bin/env python3
# vit_autoencoder_latent_visualization.py
# Pipeline using DINOv2 ViT for feature extraction + Simple Autoencoder for 16D latent space
# Now wrapped in a callable function: run_dinov2_autoencoder_preprocessing()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from collections import Counter
import pandas as pd
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import warnings

from Parking_Lot_224 import resize_parking_lot_features

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg" if you prefer and have Qt installed
import matplotlib.pyplot as plt

# -------------------------- Config --------------------------
# N_REL_CLASSES = 6
RANDOM_SEED_OFFSET = 25
VERBOSE_FLAGS = [0]

# ViT config
MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE_VIT = 32
BATCH_SIZE_AE = 128

# Autoencoder config
LATENT_DIM = 16
ENC_HIDDEN = [512, 256, 64]  # Layers for encoder
EPOCHS = 200  # ← Modifiable epochs
LR = 1e-3
VAL_SPLIT = 0.1
PATIENCE = 15

# Visualization config
REDUCTION_METHOD = "tsne"
N_COMPONENTS = 2
PERPLEXITY = 50
MAX_ITER = 1000
RANDOM_SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# -------------------------- Load Resized Data --------------------------
def load_resized_parking_lot(N_REL_CLASSES, VERBOSE_FLAGS, seed):
    # Where we expect the resized features + labels to live
    features_path = "Datasets/Parking_Lot_Data/resized_features_224.pkl"
    labels_path   = "Datasets/Parking_Lot_Data/labels.csv"

    # If resized features are missing, try to create them from the original 128x128 features
    if not os.path.exists(features_path):
        original_features_path = "Datasets/Parking_Lot_Data/features.pkl"

        if os.path.exists(original_features_path):
            print(
                f"Resized features not found at {features_path}.\n"
                f"Found original features at {original_features_path}. "
                f"Running resize_parking_lot_features(...) to generate resized features..."
            )
            resize_parking_lot_features(
                input_path=original_features_path,
                output_path=features_path,
            )
        else:
            raise FileNotFoundError(
                "Neither resized features nor original features were found.\n"
                f"Expected one of:\n"
                f"  - {features_path} (resized 224x224 features)\n"
                f"  - {original_features_path} (original 128x128 features)"
            )

    # Labels must exist; we can't auto-generate them
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels CSV not found at {labels_path}. Expected a 'label' column."
        )

    # At this point, resized features and labels are guaranteed to exist
    print("Loading resized features and labels...")
    with open(features_path, "rb") as f:
        X_images = pickle.load(f)  # (N, 224, 224, 3)

    labels_df = pd.read_csv(labels_path)
    y = labels_df["label"].tolist()

    assert len(X_images) == len(y), "Feature-label length mismatch"

    # Count occurrences of each class
    label_counts = Counter(y)

    # Sort classes by frequency: most common → least common
    sorted_counts = label_counts.most_common()
    sorted_labels = [lbl for lbl, cnt in sorted_counts]

    # Select the N_REL_CLASSES rarest classes as relevant (for viz only)
    relevant_labels = sorted_labels[-N_REL_CLASSES:]
    relevant_set = set(relevant_labels)

    # Build y_w_rel
    y_w_rel = [(label, label in relevant_set) for label in y]

    # Sparsity levels (proportion of each class, most → least common)
    total = len(y)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in sorted_counts]

    # Verbose output
    if 0 in VERBOSE_FLAGS:
        print(f"\nRunning on Resized Parking Lot dataset with {len(y)} events")
        print(f"Total classes found: {len(label_counts)}")
        print("Class distribution (top 10 most common):")
        for lbl, cnt in sorted_counts[:10]:
            print(f"   {lbl:20s} : {cnt:4d} ({cnt/total*100:5.2f}%)")
        print(
            f"\n→ Selected {N_REL_CLASSES} rarest classes as RELEVANT (for visualization):"
        )
        for lbl in relevant_labels:
            cnt = label_counts[lbl]
            print(f"   • {lbl:20s} : {cnt:4d} samples")
        print()

    return X_images, y_w_rel, sparsity_levels, relevant_labels


# -------------------------- Dataset & Feature Extraction --------------------------
class ImageDataset(Dataset):
    def __init__(self, images, processor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}


def extract_vit_features(dataloader, model, device):
    features_list = []
    model.eval()
    total = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values)
            batch_features = outputs.last_hidden_state[:, 0, :]
            features_list.append(batch_features.cpu().numpy())
            if i % 10 == 0 or i == total:
                print(f"  Processed batch {i}/{total}")
    return np.vstack(features_list)


# -------------------------- Simple Autoencoder for 16D Latent --------------------------
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_hidden):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in enc_hidden:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(enc_hidden):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


# -------------------------- Main Pipeline Function --------------------------
def run_dinov2_autoencoder_preprocessing(
    out_dir="Parking_Lot_Data",
    do_viz=True,
):
    """
    Run the full DINOv2 + Autoencoder pipeline and save:
      - {out_dir}/PARKING_LOT_DINO_latents_16d.npy
      - {out_dir}/labels.csv

    If do_viz is True, also run t-SNE / PCA and show the visualization.
    """

    # Reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # ------------------ Load resized parking lot data ------------------
    X_images, y_w_rel, sparsity_levels, rel_classes = load_resized_parking_lot(
        N_REL_CLASSES=N_REL_CLASSES,
        VERBOSE_FLAGS=VERBOSE_FLAGS,
        seed=RANDOM_SEED_OFFSET,
    )
    print(f"Dataset loaded: {len(X_images)} samples")

    # ------------------ Load DINOv2 model ------------------
    print(f"\nLoading DINOv2 model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model_vit = AutoModel.from_pretrained(MODEL_NAME)
    model_vit.eval()
    model_vit.to(device)

    # Dataset + DataLoader
    dataset = ImageDataset(X_images, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_VIT,
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
    )

    # ------------------ Extract ViT features ------------------
    print("Extracting DINOv2 features...")
    vit_features = extract_vit_features(dataloader, model_vit, device)
    print(f"ViT features shape: {vit_features.shape}")

    # Normalize features (z-score)
    X = vit_features.astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    print(f"Normalized features shape: {X.shape}")

    # ------------------ Train Autoencoder ------------------
    N = X.shape[0]
    indices = np.random.RandomState(RANDOM_SEED).permutation(N)
    train_idx, val_idx = indices[int(N * VAL_SPLIT) :], indices[: int(N * VAL_SPLIT)]

    X_train = torch.from_numpy(X[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()

    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE_AE, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=BATCH_SIZE_AE, shuffle=False)

    input_dim = X.shape[1]
    model_ae = SimpleAutoencoder(input_dim, LATENT_DIM, ENC_HIDDEN)

    optimizer = optim.Adam(model_ae.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\nTraining Autoencoder for {LATENT_DIM}D latent space over {EPOCHS} epochs...")

    best_val_loss = float("inf")
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        # ----- Training -----
        model_ae.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z, x_rec = model_ae(batch)
            loss = criterion(x_rec, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        # ----- Validation -----
        model_ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z, x_rec = model_ae(batch)
                loss = criterion(x_rec, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        if epoch <= 10 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Autoencoder training complete.\n")

    # ------------------ Extract 16D latent representations ------------------
    model_ae.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        z_all, _ = model_ae(X_tensor)
        z_all = z_all.cpu().numpy()

    print(f"16D Latent space shape: {z_all.shape}")

    # ------------------ Save 16D latents + labels ------------------
    os.makedirs(out_dir, exist_ok=True)

    latents_out_path = os.path.join(out_dir, "PARKING_LOT_DINO_latents_16d.npy")
    labels_out_path = os.path.join(out_dir, "labels.csv")

    np.save(latents_out_path, z_all)
    print(f"Saved 16D DINO latents to: {latents_out_path}")

    class_labels = [label for label, _ in y_w_rel]
    labels_df = pd.DataFrame({"label": class_labels})
    labels_df.to_csv(labels_out_path, index=False)
    print(f"Saved labels to: {labels_out_path}")

    # ------------------ Optional Dimensionality Reduction + Visualization ------------------
    if do_viz:
        print(f"Applying {REDUCTION_METHOD} for visualization...")
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        if REDUCTION_METHOD.lower() == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
            reduced_features = reducer.fit_transform(z_all)
        elif REDUCTION_METHOD.lower() == "tsne":
            reducer = TSNE(
                n_components=N_COMPONENTS,
                perplexity=PERPLEXITY,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                max_iter=MAX_ITER,
            )
            reduced_features = reducer.fit_transform(z_all)
        else:
            raise ValueError("Unsupported reduction method")

        print(f"Reduced features shape: {reduced_features.shape}")

        # Prepare labels and relevance
        class_labels = [label for label, _ in y_w_rel]
        relevant_set = set(rel_classes)

        # Separate normal and relevant (rare) classes
        normal_mask = np.array([label not in relevant_set for label in class_labels])
        relevant_mask = np.array([label in relevant_set for label in class_labels])

        normal_idx = np.where(normal_mask)[0]
        relevant_idx = np.where(relevant_mask)[0]

        print(f"Normals: {len(normal_idx)} | Relevant (rare): {len(relevant_idx)}")

        plt.figure(figsize=(14, 10))

        # 1. Subsampled background (normals) - light gray
        np.random.seed(123)
        n_background = min(1200, len(normal_idx))
        bg_idx = np.random.choice(normal_idx, n_background, replace=False)
        plt.scatter(
            reduced_features[bg_idx, 0],
            reduced_features[bg_idx, 1],
            c="lightgray",
            s=20,
            alpha=0.6,
            label=f"Normal (subsampled, n={n_background})",
        )

        # 2. All relevant classes — big, colorful, edged
        from matplotlib.cm import get_cmap
        from matplotlib.lines import Line2D

        cmap = get_cmap("tab10")

        # Keep relevant labels in the order chosen earlier (rel_classes),
        # but only those that actually appear in this subset
        relevant_labels_present = set(class_labels[i] for i in relevant_idx)
        unique_relevant_labels = [
            lab for lab in rel_classes if lab in relevant_labels_present
        ]

        # Map each relevant label -> a fixed color
        label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_relevant_labels)}

        # Color array for each relevant point
        colors = [label_to_color[class_labels[i]] for i in relevant_idx]

        plt.scatter(
            reduced_features[relevant_idx, 0],
            reduced_features[relevant_idx, 1],
            c=colors,
            s=140,
            edgecolor="black",
            linewidth=1.2,
            label="Relevant (Rare Classes)",
            zorder=5,
        )

        # Custom legend only for relevant types, with matching colors
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=lab,
                markerfacecolor=label_to_color[lab],
                markersize=12,
                markeredgecolor="black",
            )
            for lab in unique_relevant_labels
        ]

        plt.legend(
            handles=legend_elements,
            title="Relevant Class Type",
            loc="upper left",
            fontsize=10,
        )

        plt.title(
            f"DINOv2 + Autoencoder {LATENT_DIM}D Latent Space ({REDUCTION_METHOD.upper()}) — Parking Lot Dataset (224x224)\n"
            f"Total: {len(X_images)} samples | Relevant highlighted: {len(relevant_idx)}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel(f"{REDUCTION_METHOD.upper()} Component 1")
        plt.ylabel(f"{REDUCTION_METHOD.upper()} Component 2")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()  # opens a new window with the TkAgg backend

    return latents_out_path, labels_out_path


# -------------------------- Script Entry Point --------------------------
if __name__ == "__main__":
    # When run directly: generate files and visualize
    run_dinov2_autoencoder_preprocessing(
        out_dir="Datasets/Parking_Lot_Data",
        do_viz=True,
    )
