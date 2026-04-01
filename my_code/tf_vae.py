# vae_model.py ATTENTION we use dummy memeory in TF decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers."""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (Batch, SeqLen, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerVAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=10, num_heads=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.input_proj_encoder = nn.Linear(input_dim, input_dim)  # Keep dim same for transformer
        self.pos_encoder = PositionalEncoding(input_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            batch_first=True  # Important: (Batch, Seq, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Project encoder output to latent space (mu and logvar)
        # We pool the sequence to get a single representation for the latent space
        self.fc_mu = nn.Linear(input_dim * 3, latent_dim)  # 3 tokens * input_dim
        self.fc_logvar = nn.Linear(input_dim * 3, latent_dim)
        
        # --- Decoder ---
        # Project latent back to transformer dimension
        self.latent_proj_decoder = nn.Linear(latent_dim, input_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection to reconstruct original tokens
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def encode(self, x):
        """
        Encode input tokens to latent mu and logvar.
        Args:
            x: (Batch, SeqLen=2, input_dim)
        Returns:
            mu: (Batch, latent_dim)
            logvar: (Batch, latent_dim)
        """
        # Project input
        x = self.input_proj_encoder(x)  # (Batch, 3, 768)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (Batch, 3, 768)
        
        # Flatten sequence for latent projection
        encoded_flat = encoded.view(encoded.size(0), -1)  # (Batch, 3*768)
        print(encoded_flat.shape)
        
        # Get mu and logvar
        mu = self.fc_mu(encoded_flat)  # (Batch, 10)
        logvar = self.fc_logvar(encoded_flat)  # (Batch, 10)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent z back to original token space.
        Args:
            z: (Batch, latent_dim)
        Returns:
            recon: (Batch, 3, input_dim)
        """
        # Project latent to transformer dimension
        z_proj = self.latent_proj_decoder(z)  # (Batch, 768)
        
        # Expand to sequence length (2 tokens)
        # We repeat the latent vector for each token position
        z_seq = z_proj.unsqueeze(1).repeat(1, 3, 1)  # (Batch, 3, 768)
        z_seq = self.pos_encoder(z_seq)
        
        # Create a dummy memory from the latent (for cross-attention)
        memory = z_proj.unsqueeze(1)  # (Batch, 1, 768)  
        
        # Transformer decoding
        decoded = self.transformer_decoder(z_seq, memory)  # (Batch, 3, 768)
        
        # Output projection
        recon = self.output_proj(decoded)  # (Batch, 3, 768)
        
        return recon
    
    def forward(self, x):
        """
        Full VAE forward pass.
        Args:
            x: (Batch, 3, input_dim) - concatenated tokens from embedder
        Returns:
            mu: (Batch, latent_dim)
            logvar: (Batch, latent_dim)
            recon: (Batch, 3, input_dim) - reconstructed tokens
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return mu, logvar, recon
    
    def compute_kl_loss(self, mu, logvar):
        """Compute KL divergence loss."""
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
        # pass
        # return None
    
    def compute_recon_loss(self, recon, target):
        """Compute reconstruction loss (MSE)."""
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        return recon_loss

def test_vae():
    """Test function to verify VAE shapes and execution."""
    print("--- Starting Transformer VAE Test ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize VAE
    vae = TransformerVAE(
        input_dim=768, 
        latent_dim=10, 
        num_heads=8, 
        num_encoder_layers=3, 
        num_decoder_layers=3
    ).to(device)
    
    print(f"VAE loaded on {device}")
    print(f"Total parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Create dummy input (matching embedder output)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 768).to(device)
    
    # Forward pass
    mu, logvar, recon = vae(dummy_input)
    
    # Verify shapes
    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Mu Shape: {mu.shape}")
    print(f"Logvar Shape: {logvar.shape}")
    print(f"Reconstruction Shape: {recon.shape}")
    
    expected_mu_shape = (batch_size, 10)
    expected_recon_shape = (batch_size, 3, 768)
    
    success = True
    if mu.shape == expected_mu_shape:
        print(f"\n[SUCCESS] Mu shape matches expected {expected_mu_shape}")
    else:
        print(f"\n[FAIL] Mu shape {mu.shape} != expected {expected_mu_shape}")
        success = False
        
    if logvar.shape == expected_mu_shape:
        print(f"[SUCCESS] Logvar shape matches expected {expected_mu_shape}")
    else:
        print(f"[FAIL] Logvar shape {logvar.shape} != expected {expected_mu_shape}")
        success = False
        
    if recon.shape == expected_recon_shape:
        print(f"[SUCCESS] Reconstruction shape matches expected {expected_recon_shape}")
    else:
        print(f"[FAIL] Reconstruction shape {recon.shape} != expected {expected_recon_shape}")
        success = False
    
    # # Compute losses
    # kl_loss = vae.compute_kl_loss(mu, logvar)
    # recon_loss = vae.compute_recon_loss(recon, dummy_input)
    
    # print(f"\nKL Loss: {kl_loss.item():.4f}")
    # print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    
    # # Test gradient flow
    # vae.train()
    # total_loss = recon_loss + 0.001 * kl_loss  # Typical VAE loss weighting
    # total_loss.backward()
    
    # has_gradients = any(p.grad is not None for p in vae.parameters())
    # print(f"\nGradients computed: {has_gradients}")
    
    # if success and has_gradients:
    #     print("\n=== ALL TESTS PASSED ===")
    # else:
    #     print("\n=== SOME TESTS FAILED ===")
    
    return mu, logvar, recon

if __name__ == "__main__":
    test_vae()