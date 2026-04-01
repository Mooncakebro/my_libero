"""
Task Encoder with Frozen CLIP Embedder + VAE Architecture
Architecture:
- Embedder: Frozen CLIP ViT-B/32 → 512d language embedding
- VAE Encoder: (512+9) → 400 → 400 → μ(10) + logvar(10)
- Reparameterize → z(10)
- VAE Decoder: 10 → 400 → 400 → 400 → 512 (reconstruct CLIP embedding)
- Dynamics Decoder: (10+9+7) → 400 → 400 → 9 (predict next robot state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Optional


class TaskEncoder(nn.Module):
    def __init__(
        self,
        clip_path: str = "../ckpts/clip-vit-base-patch32",
        latent_dim: int = 10,
        robot_state_dim: int = 9,
        action_dim: int = 7,
        freeze_clip: bool = True
    ):
        """
        Initialize Task Encoder.
        
        Args:
            clip_path: Path to local CLIP checkpoint directory
            latent_dim: Dimension of VAE latent space (default: 10)
            robot_state_dim: Dimension of robot state vector (default: 9)
            action_dim: Dimension of action vector (default: 7)
            freeze_clip: Whether to freeze CLIP weights (default: True)
        """
        super(TaskEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.robot_state_dim = robot_state_dim
        self.action_dim = action_dim
        self.clip_embed_dim = 512  # CLIP ViT-B/32 text embedding dimension
        
        # ========== Frozen CLIP Text Encoder (Embedder) ==========
        print(f"[TaskEncoder] Loading CLIP from: {clip_path}")
        self.clip_model = CLIPModel.from_pretrained(clip_path)

        print(f"[TaskEncoder] Loading CLIP processor from: {clip_path}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        
        if freeze_clip:
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("[TaskEncoder] CLIP weights frozen ❄️")
        
        # ========== VAE Encoder: (512+9) -> 400 -> 400 -> [μ, logvar](10) ==========
        encoder_input_dim = self.clip_embed_dim + robot_state_dim
        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_input_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(400, latent_dim)
        self.logvar_head = nn.Linear(400, latent_dim)
        
        # ========== VAE Decoder: 10 -> 400 -> 400 -> 400 -> 512 ==========
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, self.clip_embed_dim),
        )
        
        # ========== Dynamics Decoder: (10+9+7) -> 400 -> 400 -> 9 ==========
        dynamics_input_dim = latent_dim + robot_state_dim + action_dim
        self.dynamics_decoder = nn.Sequential(
            nn.Linear(dynamics_input_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, robot_state_dim),
        )
    
    # ========== CLIP Language Embedding ==========
    @torch.no_grad()
    def embed_language(self, text_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode language instructions using frozen CLIP text encoder.
        
        Args:
            text_list: List of language instruction strings
            device: Torch device
            
        Returns:
            text_embeddings: (batch_size, 512) normalized CLIP embeddings
        """
        if isinstance(text_list, str):
            text_list = [text_list]
            
        text_inputs = self.clip_processor(
            text=text_list,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        # Extract text features (only need input_ids and attention_mask)
        embeddings = self.clip_model.get_text_features(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )

        return embeddings  # (batch, 512)
    
    # ========== VAE Encoder Forward ==========
    def _encode_vae(self, clip_embedding: torch.Tensor, robot_state: torch.Tensor) -> tuple:
        """
        Internal: VAE encoder forward pass.
        
        Args:
            clip_embedding: (B, 512) from CLIP
            robot_state: (B, 9) current robot state
            
        Returns:
            mu, logvar, z: All (B, latent_dim)
        """
        # Concatenate inputs
        x = torch.cat([clip_embedding, robot_state], dim=-1)  # (B, 777)
        
        # Shared encoder layers
        h = self.encoder_fc(x)  # (B, 400)
        
        # Distribution parameters
        mu = self.mu_head(h)        # (B, 10)
        logvar = self.logvar_head(h)  # (B, 10)
        
        # Reparameterization trick: z = μ + σ*ε, ε~N(0,1)
        z = self._reparameterize(mu, logvar)
        
        return mu, logvar, z
    
    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from N(0,1)
        return mu + std * eps
    
    # ========== Decoders ==========
    def _decode_language(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct CLIP embedding from latent z."""
        return self.vae_decoder(z)  # (B, 512)
    
    def _decode_dynamics(
        self, 
        z: torch.Tensor, 
        robot_state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next robot state from [z, state, action]."""
        x = torch.cat([z, robot_state, action], dim=-1)  # (B, 26)
        return self.dynamics_decoder(x)  # (B, 9)
    
    # ========== Main Forward Pass ==========
    def forward(
        self,
        text_list: List[str],
        robot_state: torch.Tensor,
        action: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the task encoder.
        
        Args:
            text_list: List of language instructions (batch size N)
            robot_state: Current robot state tensor (N, 9)
            action: Current action tensor (N, 7)
            device: Torch device (auto-detected if None)
            
        Returns:
            Dictionary containing:
                - 'mu', 'logvar', 'z': VAE latent variables
                - 'clip_embedding': Original CLIP embedding (for loss)
                - 'reconstructed_clip': Reconstructed CLIP embedding
                - 'next_state_pred': Predicted next robot state
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Step 1: Language → CLIP embedding (frozen)
        clip_embedding = self.embed_language(text_list, device)  # (B, 512)
        
        # Step 2: [CLIP + state] → VAE encode → z
        mu, logvar, z = self._encode_vae(clip_embedding, robot_state)
        
        # Step 3: Decode language embedding (VAE reconstruction)
        reconstructed_clip = self._decode_language(z)
        
        # Step 4: Decode dynamics (next state prediction)
        next_state_pred = self._decode_dynamics(z, robot_state, action)
        
        return {
            'mu': mu,                      # (B, 10)
            'logvar': logvar,              # (B, 10)
            'z': z,                        # (B, 10)
            'clip_embedding': clip_embedding,    # (B, 512) - target for recon
            'reconstructed_clip': reconstructed_clip,  # (B, 512)
            'next_state_pred': next_state_pred,      # (B, 9)
        }

    # ========== Weighted KL Div Loss Computation ==========
    def _compute_weighted_kl_div(self):
        return 0

    # ========== Loss Computation ==========
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        next_state_target: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> tuple:
        """
        Compute composite loss: VAE recon + KL + dynamics prediction.
        
        Args:
            outputs: Dict from forward()
            next_state_target: Ground truth next state (B, 9)
            weights: Loss weights {'recon', 'kl', 'dynamics'}
            
        Returns:
            total_loss, loss_dict (for logging)
        """
        if weights is None:
            weights = {'recon': 1.0, 'kl': 0.1, 'dynamics': 1.0, 'wkl': 1e-3}
        
        # 1. VAE Reconstruction Loss (MSE on CLIP embedding space)
        recon_loss = F.mse_loss(
            outputs['reconstructed_clip'],
            outputs['clip_embedding'],
            reduction='mean'
        )
        
        # 2. KL Divergence Loss (analytical form for diagonal Gaussian)
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1 + outputs['logvar'] 
                - outputs['mu'].pow(2) 
                - outputs['logvar'].exp(), 
                dim=1
            )
        )
        
        # 3. Dynamics Prediction Loss (MSE on next state)
        dynamics_loss = F.mse_loss(
            outputs['next_state_pred'],
            next_state_target,
            reduction='mean'
        )

        # 4. Weighted KL Divergence Loss (analytical form for DPMM)
        w_kl_loss = self._compute_weighted_kl_div()
        
        # Weighted total
        total_loss = (
            weights['recon'] * recon_loss +
            weights['kl'] * kl_loss +
            weights['wkl'] * w_kl_loss +
            weights['dynamics'] * dynamics_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'wkl': w_kl_loss.item(),
            'dynamics': dynamics_loss.item()
        }
        
        return total_loss, loss_dict
    
    # ========== Utility: Sample from Prior ==========
    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample z from standard normal prior N(0, I)."""
        return torch.randn(batch_size, self.latent_dim, device=device)