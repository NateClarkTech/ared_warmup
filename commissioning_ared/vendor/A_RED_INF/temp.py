# train_parking_dagmm_best.py
# Configurable PCA version – easily toggle PCA on/off

from DAGMM import DAGMM
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.decomposition import PCA
import warnings
from umap import UMAP

warnings.filterwarnings('ignore')

# -------------------------- Config --------------------------
N_REL_CLASSES = 8
RANDOM_SEED = 42
MAX_EPOCHS = 300
PATIENCE = 150
BATCH_SIZE = 512
N_COMPONENTS = 12
LATENT_DIM = 64

# NEW: Toggle PCA usage here!
USE_PCA = False           # Set to True to enable PCA, False to skip it entirely
PCA_COMPS = 1024          # Only used if USE_PCA=True (e.g., 512, 1024, or None for all components)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------- 1. Load Data --------------------------
from main_helper_functions import parking_lot_setup_for_main

print(f"Loading Parking Lot data with N_REL_CLASSES = {N_REL_CLASSES}")
X_raw, y_w_rel, sparsity_levels, relevant_labels = parking_lot_setup_for_main(
    N_REL_CLASSES=N_REL_CLASSES,
    VERBOSE_FLAGS=[0],
    seed=RANDOM_SEED
)

# X_raw is already flattened or in correct shape? We'll assume it's ready or flatten safely
if X_raw.ndim > 2:
    print(f"Flattening images from {X_raw.shape} → ", end="")
    X_flat = X_raw.reshape(len(X_raw), -1).astype(np.float32)
    print(f"{X_flat.shape}")
else:
    X_flat = X_raw.astype(np.float32)

# -------------------------- Optional PCA --------------------------
if USE_PCA:
    if PCA_COMPS is None:
        PCA_COMPS = min(X_flat.shape)  # retain all components

    print(f"Applying PCA(n_components={PCA_COMPS}, whiten=False, random_state={RANDOM_SEED})...")
    pca = PCA(n_components=PCA_COMPS, random_state=RANDOM_SEED, svd_solver='full')
    X_pca = pca.fit_transform(X_flat)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA retained {explained_variance*100:.2f}% of total variance with {PCA_COMPS} components")

    if explained_variance < 0.90:
        print(f"   WARNING: Only {explained_variance*100:.1f}% variance retained! Consider increasing PCA_COMPS.")
else:
    print("Skipping PCA (USE_PCA=False)")
    X_pca = X_flat
    explained_variance = 1.0  # 100% retained

# -------------------------- Scaling to [-1, 1] --------------------------
X_min = X_pca.min(axis=0)
X_max = X_pca.max(axis=0)
# Avoid division-safe scaling
X_full = (X_pca - X_min) / (X_max - X_min + 1e-8)
X_full = X_full * 2.0 - 1.0

input_dim = X_full.shape[1]
print(f"Final preprocessed data: {X_full.shape[0]:,} samples × {input_dim} dimensions")
if USE_PCA:
    print(f"   (after PCA + scaling, {explained_variance*100:.2f}% variance retained)")
else:
    print(f"   (no PCA, full original features used)")

# ==================== Extract labels ====================
raw_labels = np.array([y[0] for y in y_w_rel])
relevance_flags = np.array([y[1] for y in y_w_rel])

# -------------------------- Train/Val Split --------------------------
val_ratio = 0.1
n_val = int(len(X_full) * val_ratio)
indices = np.random.RandomState(RANDOM_SEED).permutation(len(X_full))
train_idx, val_idx = indices[:-n_val], indices[-n_val:]

X_train = torch.from_numpy(X_full[train_idx])
X_val = torch.from_numpy(X_full[val_idx])

train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)

# -------------------------- Model --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on {device}")
print(f"Hyperparams → latent_dim={LATENT_DIM}, GMM components={N_COMPONENTS}, PCA={'ON' if USE_PCA else 'OFF'}")

model = DAGMM(
    input_dim=input_dim,
    latent_dim=LATENT_DIM,
    n_components=N_COMPONENTS,
    lambda_energy=0.1,
    lambda_cov=0.005,
    enc_hidden=[2048, 1024, 512, 256, 128],
    est_hidden=[256, 128, 64],
    dropout=0.3,
    activation="tanh",
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

# -------------------------- Training Loop (unchanged) --------------------------
# ... [rest of your training loop remains 100% identical] ...
# (I'm keeping it collapsed for brevity — you can paste your original loop here unchanged)

# For completeness, here’s the key part again:
best_val_energy = float('inf')
patience_counter = 0

print("\nStarting training with OPTIMIZED DaGMM...\n")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    # Full-batch GMM parameter estimation
    z_collect, gamma_collect = [], []
    with torch.no_grad():
        for (x,) in train_loader:
            x = x.to(device)
            _, _, z, gamma = model(x)
            z_collect.append(z.cpu())
            gamma_collect.append(gamma.cpu())
    z_all = torch.cat(z_collect).to(device)
    gamma_all = torch.cat(gamma_collect).to(device)
    model.compute_gmm_params(z_all, gamma_all)

    # Training step
    epoch_loss = 0.0
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        z_c, x_rec, z, gamma = model(x)
        loss_dict = model.loss_function(x, x_rec, z, gamma)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss_dict["loss"].item() * x.size(0)
    epoch_loss /= len(train_loader.dataset)

    # Validation energy
    model.eval()
    with torch.no_grad():
        val_z = []
        for (x,) in val_loader:
            x = x.to(device)
            _, _, z, _ = model(x)
            val_z.append(z.cpu())
        val_z = torch.cat(val_z).to(device)
        val_energy = model.compute_energy(val_z).mean().item()

    if epoch <= 10 or epoch % 30 == 0 or epoch == MAX_EPOCHS:
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Val Energy: {val_energy:.4f}")

    if val_energy < best_val_energy - 1e-4:
        best_val_energy = val_energy
        patience_counter = 0
        torch.save(model.state_dict(), f"results/dagmm_best_NREL{N_REL_CLASSES}.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Load best model and do inference (unchanged from your original)
model.load_state_dict(torch.load(f"results/dagmm_best_NREL{N_REL_CLASSES}.pth"))

print("\nFinal inference...")
model.eval()
with torch.no_grad():
    X_tensor = torch.from_numpy(X_full).to(device)
    z_c_all, _, z_all, _ = model(X_tensor)

    z_c_all = z_c_all.cpu().numpy()
    z_all = z_all.cpu().numpy()

    anomaly_scores = model.predict_energy(X_full)
    if torch.is_tensor(anomaly_scores):
        anomaly_scores = anomaly_scores.cpu().numpy()

# Save results + visualization (same as before)
os.makedirs("results", exist_ok=True)

with open(f"results/preprocessed_X_latent_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(z_c_all, f)
with open(f"results/preprocessed_X_full_z_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(z_all, f)
with open(f"results/y_w_rel_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(y_w_rel, f)
with open(f"results/dagmm_anomaly_scores_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(anomaly_scores, f)

top_idx = np.argsort(anomaly_scores)[-100:][::-1]
rel_count = relevance_flags[top_idx].sum()

print("Latent representations saved!")
print(f"   → z_c_all shape       : {z_c_all.shape}")
print(f"   → z_all shape         : {z_all.shape}    ← RECOMMENDED FOR A/RED")
print(f"   → Top-100 recall      : {rel_count}/100 ({rel_count / 100:.1%}) relevant rare classes")

# ... rest of UMAP visualization code unchanged ...