# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) for semantic feature compression in IoT data.
Replaces hand-crafted feature extraction with learned latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input → hidden → mu, log_var
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: latent → hidden → output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # output in [0,1] (data must be pre-normalized)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    @torch.no_grad()
    def get_latent(self, x):
        self.eval()
        mu, _ = self.encode(x)
        return mu


def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    """ELBO loss = reconstruction MSE + beta * KL divergence."""
    recon = F.mse_loss(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon + beta * kl
    return total, recon.item(), kl.item()


def train_vae(X_np, input_dim, latent_dim=16, hidden_dim=64,
              epochs=100, lr=1e-3, batch_size=256, beta=1.0,
              device=None, verbose=True):
    """
    Train VAE on a numpy array X_np (already normalized to [0,1]).
    Returns: trained VAE, list of per-epoch losses.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    vae = VAE(input_dim, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'total': [], 'recon': [], 'kl': []}

    for epoch in range(epochs):
        vae.train()
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, log_var = vae(batch)
            loss, r, k = vae_loss(x_recon, batch, mu, log_var, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            epoch_total += loss.item()
            epoch_recon += r
            epoch_kl += k

        scheduler.step()
        n = len(loader)
        history['total'].append(epoch_total / n)
        history['recon'].append(epoch_recon / n)
        history['kl'].append(epoch_kl / n)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  VAE [{epoch+1}/{epochs}] "
                  f"Loss={epoch_total/n:.2f}  "
                  f"Recon={epoch_recon/n:.2f}  "
                  f"KL={epoch_kl/n:.2f}")

    return vae, history


@torch.no_grad()
def encode_dataset(vae, X_np, device=None, batch_size=512):
    """Encode full dataset → latent mu vectors [N, latent_dim]."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.eval()
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    latents = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        mu, _ = vae.encode(batch)
        latents.append(mu.cpu().numpy())
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def reconstruction_error(vae, X_np, device=None, batch_size=512):
    """Return per-sample MSE reconstruction error."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.eval()
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    errors = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        x_recon, _, _ = vae(batch)
        mse = F.mse_loss(x_recon, batch, reduction='none').mean(dim=1)
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors, axis=0)
