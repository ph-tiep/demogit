# -*- coding: utf-8 -*-
"""Variational Autoencoder for semantic feature compression"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

class VAE(nn.Module):
    """Variational Autoencoder for data compression"""
    
    def __init__(self, input_dim, latent_dim=3):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Latent space
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, input_dim)
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def compress(self, x):
        """Compress data to latent representation"""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss, recon_loss.item(), kl_loss.item()

def train_vae(data, input_dim, latent_dim=3, epochs=50, batch_size=64, lr=1e-3):
    """
    Train VAE on semantic features
    
    Args:
        data: numpy array of features
        input_dim: dimension of input features
        latent_dim: dimension of latent space (compressed)
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        
    Returns:
        trained VAE model, scaler, training losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Convert to tensor
    data_tensor = torch.FloatTensor(data_normalized).to(device)
    
    # Create VAE model
    vae = VAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    # Training
    vae.train()
    losses = []
    recon_losses = []
    kl_losses = []
    
    n_samples = len(data_tensor)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Training VAE: {input_dim}D → {latent_dim}D compression")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batch = data_tensor[batch_indices]
            
            # Forward pass
            recon_batch, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss
            epoch_kl += kl_loss
        
        # Average losses
        avg_loss = epoch_loss / n_samples
        avg_recon = epoch_recon / n_samples
        avg_kl = epoch_kl / n_samples
        
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} "
                  f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
    
    vae.eval()
    print("VAE training completed!")
    
    return vae, scaler, losses, recon_losses, kl_losses

def compress_data_with_vae(data, vae, scaler):
    """
    Compress data using trained VAE
    
    Args:
        data: numpy array of features
        vae: trained VAE model
        scaler: fitted StandardScaler
        
    Returns:
        compressed latent representations
    """
    device = next(vae.parameters()).device
    
    # Normalize
    data_normalized = scaler.transform(data)
    data_tensor = torch.FloatTensor(data_normalized).to(device)
    
    # Compress
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(data_tensor)
        latent = mu.cpu().numpy()
    
    return latent

def reconstruct_data_with_vae(latent_data, vae, scaler):
    """
    Reconstruct data from latent representations
    
    Args:
        latent_data: numpy array of latent representations
        vae: trained VAE model
        scaler: fitted StandardScaler
        
    Returns:
        reconstructed data in original space
    """
    device = next(vae.parameters()).device
    
    # Convert to tensor
    latent_tensor = torch.FloatTensor(latent_data).to(device)
    
    # Decode
    vae.eval()
    with torch.no_grad():
        recon_normalized = vae.decode(latent_tensor).cpu().numpy()
    
    # Inverse transform
    recon_original = scaler.inverse_transform(recon_normalized)
    
    return recon_original
