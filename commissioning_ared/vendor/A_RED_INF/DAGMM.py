import math
import os
from typing import Optional, Tuple, Union, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from main_helper_functions import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns  # Added for consistent color palette

class DAGMM(nn.Module):
    """
    Clean, modern DaGMM implementation.

    - Default architecture follows typical encoder/decoder widths used in the paper.
    - fit(...) supports early stopping, validation split, GPU, and stable GMM estimation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        n_components: int = 16,
        lambda_energy: float = 0.1,
        lambda_cov: float = 0.005,
        enc_hidden: Optional[list] = [128, 64],
        est_hidden: Optional[list] = [64, 128],
        dropout: float = 0.5,
        activation: str = "tanh",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)
        self.lambda_energy = float(lambda_energy)
        self.lambda_cov = float(lambda_cov)
        self.dropout = float(dropout)

        # Default architectures (paper-style)
        if enc_hidden is None:
            enc_hidden = [60, 30, 10]   # encoder widths
        if est_hidden is None:
            est_hidden = [10]

        self.activation = self._get_activation(activation)

        # ---------------- Encoder ----------------
        enc_layers = []
        prev = self.input_dim
        for h in enc_hidden:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(self.activation)
            prev = h
        enc_layers.append(nn.Linear(prev, self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ---------------- Decoder ----------------
        dec_layers = []
        prev = self.latent_dim
        for h in reversed(enc_hidden): # hi
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(self.activation)
            prev = h
        dec_layers.append(nn.Linear(prev, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # ---------------- Estimation Network ----------------
        est_input_dim = self.latent_dim + 2  # z_c , euclid , cosine
        est_layers = []
        prev = est_input_dim
        for h in est_hidden:
            est_layers.append(nn.Linear(prev, h))
            est_layers.append(self.activation)
            prev = h
        est_layers.append(nn.Dropout(self.dropout))
        est_layers.append(nn.Linear(prev, self.n_components))
        est_layers.append(nn.Softmax(dim=1))
        self.estimation = nn.Sequential(*est_layers)

        # GMM buffers (CPU/GPU safe)
        self.register_buffer("phi", torch.zeros(self.n_components))
        self.register_buffer("mu", torch.zeros(self.n_components, est_input_dim))
        self.register_buffer("cov", torch.zeros(self.n_components, est_input_dim, est_input_dim))

        self.to(self.device)

    def _get_activation(self, name: str):
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "softplus":
            return nn.Softplus()
        return nn.Tanh()

    # ============================================================
    # Forward helpers
    # ============================================================
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def relative_euclidean_dist(self, x: Tensor, x_rec: Tensor) -> Tensor:
        num = torch.norm(x - x_rec, p=2, dim=1, keepdim=True)
        denom = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return num / denom

    def cosine_similarity(self, x: Tensor, x_rec: Tensor) -> Tensor:
        return F.cosine_similarity(x, x_rec, dim=1, eps=1e-8).unsqueeze(1)

    def forward(self, x: Tensor):
        """
        Returns: z_c , x_rec , z=[z_c, euc, cos], gamma
        """
        x = x.to(self.device)
        z_c = self.encode(x)
        x_rec = self.decode(z_c)

        euc = self.relative_euclidean_dist(x, x_rec)
        cos = self.cosine_similarity(x, x_rec)
        z = torch.cat([z_c, euc, cos], dim=1)

        gamma = self.estimation(z)
        return z_c, x_rec, z, gamma

    # ============================================================
    # GMM / Energy
    # ============================================================
    def compute_gmm_params(self, z, gamma, reg_covar=1e-6):
        z = z.detach()
        gamma = gamma.detach()
        N, D = z.shape
        K = self.n_components

        gamma_sum = gamma.sum(dim=0) + 1e-8
        phi = gamma_sum / N
        mu = (gamma.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=0) / gamma_sum.unsqueeze(-1)

        z_center = z.unsqueeze(1) - mu.unsqueeze(0)
        weighted = (
            gamma.unsqueeze(-1).unsqueeze(-1)
            * (z_center.unsqueeze(-1) * z_center.unsqueeze(-2))
        ).sum(dim=0)
        cov = weighted / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        cov = cov + reg_covar * torch.eye(D, device=cov.device).unsqueeze(0)

        self.phi.copy_(phi)
        self.mu.copy_(mu)
        self.cov.copy_(cov)
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None):
        if phi is None:
            phi, mu, cov = self.phi, self.mu, self.cov

        B, D = z.shape
        K = phi.shape[0]

        z = z.unsqueeze(1)
        diff = z - mu.unsqueeze(0)

        cov_reg = cov + 1e-6 * torch.eye(D, device=cov.device).unsqueeze(0)
        sign, logdet = torch.linalg.slogdet(cov_reg)
        if (sign <= 0).any():
            cov_reg = cov_reg + 1e-3 * torch.eye(D, device=cov.device).unsqueeze(0)
            sign, logdet = torch.linalg.slogdet(cov_reg)
        cov_inv = torch.linalg.inv(cov_reg)

        mahala = torch.einsum("bkd,kde,bke->bk", diff, cov_inv, diff)
        const = D * math.log(2 * math.pi)
        log_prob = -0.5 * (mahala + logdet.unsqueeze(0) + const)

        log_phi = torch.log(phi.unsqueeze(0).clamp_min(1e-12))
        log_sum = torch.logsumexp(log_phi + log_prob, dim=1)
        return -log_sum  # energy

    # ============================================================
    # Loss
    # ============================================================
    def loss_function(self, x, x_rec, z, gamma):
        recon_loss = F.mse_loss(x_rec, x, reduction="mean")
        energy = self.compute_energy(z)
        energy_mean = energy.mean()

        cov_diag = torch.diagonal(self.cov, dim1=-2, dim2=-1)
        cov_diag_term = torch.sum(1.0 / (cov_diag + 1e-8))

        loss = recon_loss + self.lambda_energy * energy_mean + self.lambda_cov * cov_diag_term

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "energy_mean": energy_mean,
            "cov_diag_term": cov_diag_term,
            "energy_vec": energy,
        }

    # ============================================================
    # Training Loop (fit)
    # ============================================================
    def fit(
        self,
        X,
        epochs=200,
        batch_size=1024,
        lr=1e-3,
        weight_decay=1e-6,
        val_split=0.1,
        shuffle=True,
        patience=10,
        save_best_path=None,
        verbose=True,
    ):

        # Prepare data
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        else:
            X = X.detach().float()

        N = X.shape[0]
        idxs = np.arange(N)
        if shuffle:
            np.random.shuffle(idxs)

        # train/val split
        if val_split > 0:
            split = int((1 - val_split) * N)
            train_idx, val_idx = idxs[:split], idxs[split:]
        else:
            train_idx, val_idx = idxs, np.array([], dtype=int)

        X_train = X[train_idx].to(self.device)
        X_val = X[val_idx].to(self.device) if len(val_idx) > 0 else None

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train),
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {"train_loss": [], "train_recon": [], "train_energy": [], "val_energy": []}

        # ---------------- Initialize GMM ----------------
        with torch.no_grad():
            z_all, gamma_all = [], []
            self.eval()
            train_ds = torch.utils.data.TensorDataset(X_train)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            for (batch,) in train_loader:
                _, _, z, gamma = self.forward(batch)
                z_all.append(z.cpu())
                gamma_all.append(gamma.cpu())
            z_all = torch.cat(z_all).to(self.device)
            gamma_all = torch.cat(gamma_all).to(self.device)
            self.compute_gmm_params(z_all, gamma_all)

        # ---------------- Training Loop ----------------
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):

            self.train()
            ep_loss = ep_recon = ep_energy = 0
            n_batches = 0

            # Collect reps for full GMM update
            z_collect, gamma_collect = [], []

            # Pass 1: collect z/gamma
            for (batch,) in train_loader:
                with torch.no_grad():
                    _, _, z, gamma = self.forward(batch)
                    z_collect.append(z.cpu())
                    gamma_collect.append(gamma.cpu())

            z_all = torch.cat(z_collect).to(self.device)
            gamma_all = torch.cat(gamma_collect).to(self.device)
            self.compute_gmm_params(z_all, gamma_all)

            # Pass 2: update nets
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                z_c, x_rec, z, gamma = self.forward(batch)
                loss_dict = self.loss_function(batch, x_rec, z, gamma)

                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                ep_recon += loss_dict["recon_loss"].item()
                ep_energy += loss_dict["energy_mean"].item()
                n_batches += 1

            ep_loss /= n_batches
            ep_recon /= n_batches
            ep_energy /= n_batches

            # Validation
            if X_val is not None and len(X_val) > 0:
                self.eval()
                with torch.no_grad():
                    _, _, z_val, _ = self.forward(X_val)
                    val_energy = float(self.compute_energy(z_val).mean().item())
            else:
                val_energy = float("nan")

            history["train_loss"].append(ep_loss)
            history["train_recon"].append(ep_recon)
            history["train_energy"].append(ep_energy)
            history["val_energy"].append(val_energy)

            if verbose:
                print(f"Epoch {epoch}/{epochs} | loss={ep_loss:.6f} | recon={ep_recon:.6f} | trainE={ep_energy:.6f} | valE={val_energy:.6f}")

            # Early stopping
            metric = val_energy if not math.isnan(val_energy) else ep_energy

            if metric < best_val:
                best_val = metric
                no_improve = 0
                if save_best_path:
                    self.save(save_best_path)
            else:
                no_improve += 1

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Restore best model (optional)
        if save_best_path and os.path.exists(save_best_path):
            self.load(save_best_path)

        return history

    # ============================================================
    # Prediction
    # ============================================================
    def predict_energy(self, X, batch_size=4096):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        self.eval()
        energies = []
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=batch_size)
            for (batch,) in loader:
                _, _, z, _ = self.forward(batch)
                e = self.compute_energy(z)
                energies.append(e.cpu().numpy())
        return np.concatenate(energies, axis=0)

    score_samples = predict_energy

    # ============================================================
    # Save / Load
    # ============================================================
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "input_dim": self.input_dim},
            path,
        )

    def load(self, path: str, map_location=None):
        data = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(data["state_dict"])
        self.to(self.device)


# ============================================================
# Main (example usage)
# ============================================================
if __name__ == "__main__":
    # -------------------------
    # Parameters
    # -------------------------
    DATA_SOURCE = "PARKING_LOT"
    N_REL_CLASSES = 4
    RANDOM_SEED_OFFSET = 25
    VERBOSE_FLAGS = [0]

    # DaGMM hyperparameters (tuned for 128×128 parking images)
    latent_dim = 16
    n_components = 4
    enc_hidden = [1024, 512, 256, 64]
    est_hidden = [64, 32]
    dropout = 0.3
    lambda_energy = 0.1
    lambda_cov = 0.005

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Load and preprocess data
    # -------------------------
    X_skewed, y_w_rel, sparsity_levels, rel_classes = parking_lot_setup_for_main(
        1, VERBOSE_FLAGS, RANDOM_SEED_OFFSET
    )
    print(f"Dataset loaded: {X_skewed.shape[0]} samples, labels type: {type(y_w_rel[0])}")

    # Flatten images: (N, 128, 128) → (N, 16384)
    X = X_skewed.reshape(X_skewed.shape[0], -1).astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)  # normalize to [0,1]

    # Convert labels to list of strings for easier handling
    y_w_rel = [str(label) for label in y_w_rel]

    # -------------------------
    # Initialize and train DaGMM
    # -------------------------
    model = DAGMM(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        n_components=n_components,
        enc_hidden=enc_hidden,
        est_hidden=est_hidden,
        dropout=dropout,
        lambda_energy=lambda_energy,
        lambda_cov=lambda_cov,
        device=device
    ).to(device)

    print("\nTraining DaGMM...")
    history = model.fit(
        X,
        epochs=150,
        batch_size=128,
        val_split=0.1,
        patience=15,
        lr=1e-3,
        verbose=True
    )

    print("Training complete.\n")

    # -------------------------
    # Compute anomaly scores
    # -------------------------
    scores = model.predict_energy(X)
    print(f"Energy scores computed: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

    # Top-20 anomalies
    top_k = 20
    top_indices = np.argsort(scores)[-top_k:][::-1]
    print(f"\nTop {top_k} anomalies:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:2d}. idx={idx:4d} | score={scores[idx]:8.4f} | label={y_w_rel[idx]}")

    # -------------------------
    # Extract latent representations
    # -------------------------
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        z_c, _, _, _ = model.forward(X_tensor)
        z_c = z_c.cpu().numpy()

    print(f"Latent space shape: {z_c.shape}")

    # -------------------------
    # t-SNE (with perplexity tuned for ~4k points)
    # -------------------------
    print("Running t-SNE (this may take 30–60 seconds)...")
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        init='pca',
        learning_rate='auto',
        n_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    z_2d = tsne.fit_transform(z_c)

    # -------------------------
    # Visualization: each class a different color
    # -------------------------
    plt.figure(figsize=(14, 10))

    unique_labels = sorted(set(y_w_rel))
    palette = sns.color_palette("tab20", len(unique_labels))
    color_map = {cls: palette[i] for i, cls in enumerate(unique_labels)}

    for cls in unique_labels:
        mask = np.array([label == cls for label in y_w_rel])
        color = color_map[cls]
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[color], s=60, label=f"{cls} ({mask.sum()})",
                    alpha=1.0, edgecolors='black', linewidth=1.0, zorder=5)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Class")
    plt.title("DaGMM Latent Space (t-SNE) — Parking Lot Dataset\n"
              f"Total: {len(X)} samples", fontsize=16, pad=20)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Optional: Save results
    # -------------------------
    np.savez("dagmm_results.npz",
             z_2d=z_2d,
             scores=scores,
             labels=np.array(y_w_rel),
             top_indices=top_indices)
    print("Results saved to dagmm_results.npz")