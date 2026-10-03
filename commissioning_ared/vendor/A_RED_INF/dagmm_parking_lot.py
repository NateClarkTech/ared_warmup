# dagmm_parking_lot.py
# Fixed + progress bars + robust

import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from DAGMM import DAGMM
from sklearn.decomposition import PCA
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_dagmm_features_parking_lot(
    N_REL_CLASSES=8,
    USE_PCA=False,
    PCA_COMPS=1024,
    seed=42,
    device=None,
    verbose=True
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # --------------------- Cache key ---------------------
    pca_suffix = f"{PCA_COMPS}" if USE_PCA else "OFF"
    cache_key = f"NREL{N_REL_CLASSES}_PCA{pca_suffix}"
    cache_dir = "results"
    os.makedirs(cache_dir, exist_ok=True)

    latent_path = os.path.join(cache_dir, f"preprocessed_X_full_z_{cache_key}.pkl")
    labels_path = os.path.join(cache_dir, f"y_w_rel_{cache_key}.pkl")
    scores_path = os.path.join(cache_dir, f"dagmm_scores_{cache_key}.pkl")

    if os.path.exists(latent_path) and os.path.exists(labels_path):
        if verbose:
            print(f"Loading cached DaGMM features: {cache_key}")
        with open(latent_path, "rb") as f:
            X_latent = pickle.load(f)
        with open(labels_path, "rb") as f:
            y_w_rel = pickle.load(f)
        X_latent = np.array(X_latent, dtype=np.float32)

        # Recompute sparsity & rel_classes
        labels_only = [lbl for lbl, _ in y_w_rel]
        label_counts = Counter(labels_only)
        total = len(labels_only)
        sparsity_levels = [(lbl, cnt/total) for lbl, cnt in label_counts.most_common()]
        rel_classes = sorted({lbl for lbl, is_rel in y_w_rel if is_rel})

        return X_latent, y_w_rel, sparsity_levels, rel_classes

    # --------------------- Fresh training ---------------------
    if verbose:
        print(f"Computing DaGMM features: {cache_key}")
        print(f"   → PCA={'ON' if USE_PCA else 'OFF'}, N_REL_CLASSES={N_REL_CLASSES}")

    # Load raw data
    from main_helper_functions import parking_lot_setup_for_main
    X_raw, y_w_rel, _, _ = parking_lot_setup_for_main(
        N_REL_CLASSES=N_REL_CLASSES,
        VERBOSE_FLAGS=[1] if verbose else [0],
        seed=seed
    )

    if X_raw.ndim > 2:
        X_flat = X_raw.reshape(len(X_raw), -1).astype(np.float32)
    else:
        X_flat = X_raw.astype(np.float32)

    # Optional PCA
    if USE_PCA:
        n_comps = PCA_COMPS or min(X_flat.shape)
        pca = PCA(n_components=n_comps, random_state=seed, svd_solver='full')
        X_reduced = pca.fit_transform(X_flat)
        if verbose:
            var = pca.explained_variance_ratio_.sum()
            print(f"   → PCA({n_comps}) → {X_reduced.shape[1]} dims, {var*100:.1f}% variance retained")
    else:
        X_reduced = X_flat
        if verbose:
            print(f"   → No PCA → using full {X_flat.shape[1]} dimensions")

    # Scale to [-1, 1]
    X_min, X_max = X_reduced.min(axis=0), X_reduced.max(axis=0)
    X_full = (X_reduced - X_min) / (X_max - X_min + 1e-8)
    X_full = X_full * 2.0 - 1.0
    X_full = np.array(X_full, dtype=np.float32)  # ensure numpy

    input_dim = X_full.shape[1]

    # Train/val split
    val_ratio = 0.1
    indices = np.random.RandomState(seed).permutation(len(X_full))
    n_val = int(len(X_full) * val_ratio)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    X_train_torch = torch.from_numpy(X_full[train_idx]).to(device)
    X_val_torch = torch.from_numpy(X_full[val_idx]).to(device)

    train_loader = DataLoader(TensorDataset(X_train_torch), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_torch), batch_size=512, shuffle=False)

    # Model
    model = DAGMM(
        input_dim=input_dim,
        latent_dim=64,
        n_components=12,
        lambda_energy=0.1,
        lambda_cov=0.005,
        enc_hidden=[2048, 1024, 512, 256, 128],
        est_hidden=[256, 128, 64],
        dropout=0.3,
        activation="tanh",
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    best_val_energy = float('inf')
    patience_counter = 0
    max_epochs = 300
    patience = 150

    if verbose:
        print(f"Starting DaGMM training... ({max_epochs} max epochs, early stopping)")

    for epoch in tqdm(range(1, max_epochs + 1), desc="DaGMM Training", disable=not verbose):
        model.train()

        # GMM parameter estimation (full batch)
        z_collect, gamma_collect = [], []
        with torch.no_grad():
            for (x,) in train_loader:
                x = x.to(device)
                _, _, z, gamma = model(x)
                z_collect.append(z)
                gamma_collect.append(gamma)
        z_all = torch.cat(z_collect)
        gamma_all = torch.cat(gamma_collect)
        model.compute_gmm_params(z_all, gamma_all)

        # Training step
        epoch_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            z_c, x_rec, z, gamma = model(x)
            loss_dict = model.loss_function(x, x_rec, z, gamma)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss_dict["loss"].item()

        # Validation energy
        model.eval()
        with torch.no_grad():
            val_z_list = []
            for (x,) in val_loader:
                x = x.to(device)
                _, _, z, _ = model(x)
                val_z_list.append(z)
            val_z = torch.cat(val_z_list)
            val_energy = model.compute_energy(val_z).mean().item()

        # Early stopping & checkpoint
        if val_energy < best_val_energy - 1e-4:
            best_val_energy = val_energy
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}  # CPU-safe
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break

        if epoch <= 5 or epoch % 30 == 0 or epoch == max_epochs:
            if verbose:
                print(f"   Epoch {epoch:3d} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Energy: {val_energy:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final inference on full dataset
    if verbose:
        print("Running final inference on full dataset...", flush=True)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_full).to(device)
        z_c_all, _, z_all, _ = model(X_tensor)

        # Safe way to get anomaly scores (works with any DAGMM version)
        energy = model.predict_energy(X_tensor)
        if torch.is_tensor(energy):
            anomaly_scores = energy.cpu().numpy()
        else:
            anomaly_scores = np.asarray(energy, dtype=np.float32)

    X_latent = z_all.cpu().numpy().astype(np.float32)

    # Cache everything
    with open(latent_path, "wb") as f:
        pickle.dump(X_latent, f)
    with open(labels_path, "wb") as f:
        pickle.dump(y_w_rel, f)
    with open(scores_path, "wb") as f:
        pickle.dump(anomaly_scores, f)

    if verbose:
        rel_in_top100 = np.sum(np.array([r for _, r in y_w_rel])[np.argsort(anomaly_scores)[-100:]])
        print(f"Done! Top-100 rare recall: {rel_in_top100}/100")

    # Build return values
    labels_only = [lbl for lbl, _ in y_w_rel]
    label_counts = Counter(labels_only)
    total = len(labels_only)
    sparsity_levels = [(lbl, cnt/total) for lbl, cnt in label_counts.most_common()]
    rel_classes = sorted({lbl for lbl, is_rel in y_w_rel if is_rel})

    return X_latent, y_w_rel, sparsity_levels, rel_classes