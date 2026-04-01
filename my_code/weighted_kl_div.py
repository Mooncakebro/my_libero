import torch
import torch.nn.functional as F

def compute_weighted_kl_loss(mu, log_var, anchor_mu, anchor_var, soft_labels=None, temperature=1.0):
    """
    Compute KL divergence between encoded distribution and anchor distributions.
    
    Args:
        mu: Encoded mean (batch_size, latent_dim)
        log_var: Encoded log variance (batch_size, latent_dim)
        anchor_mu: Anchor means (num_anchors, latent_dim)
        anchor_var: Anchor variances (num_anchors, latent_dim)
        soft_labels: Soft assignment probabilities (num_anchors, batch_size)
        temperature: Temperature for softmax weighting
        
    Returns:
        kl_loss: Weighted KL divergence loss
        kl_per_anchor: KL divergence per anchor (for logging)
    """
    batch_size = mu.size(0)
    num_anchors = anchor_mu.size(0)
    
    # Expand dimensions for broadcasting
    mu_q = mu.unsqueeze(1).expand(-1, num_anchors, -1)  # (B, K, D)
    log_var_q = log_var.unsqueeze(1).expand(-1, num_anchors, -1)
    var_q = torch.exp(log_var_q)
    
    mu_p = anchor_mu.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, D)
    var_p = anchor_var.unsqueeze(0).expand(batch_size, -1, -1)
    
    # KL(q||p) = 0.5 * [log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
    kl_div = 0.5 * (
        torch.log(var_p + 1e-8) - log_var_q +
        (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1
    )
    
    kl_div = kl_div.sum(dim=-1)  # (B, K)
    kl_per_anchor = kl_div.mean(dim=0).detach()  # For logging
    
    if soft_labels is not None and num_anchors > 0:
        soft_labels = soft_labels.transpose(0, 1)  # (B, K)
        if temperature != 1.0:
            soft_labels = F.softmax(soft_labels / temperature, dim=-1)
        weighted_kl = torch.sum(kl_div * soft_labels, dim=-1)
        kl_loss = weighted_kl.mean()
    else:
        kl_loss = kl_div.mean()
    
    return kl_loss, kl_per_anchor


def compute_soft_labels(mu, log_var, anchor_mu, anchor_var):
    """
    Compute soft assignment probabilities based on likelihood of z belonging to each anchor.
    
    Returns:
        soft_labels: (num_anchors, batch_size) - probability each sample belongs to each anchor
    """
    batch_size = mu.size(0)
    num_anchors = anchor_mu.size(0)
    
    if num_anchors == 0:
        return None
    
    # Compute negative KL as log-likelihood proxy
    mu_q = mu.unsqueeze(1).expand(-1, num_anchors, -1)
    var_q = torch.exp(log_var).unsqueeze(1).expand(-1, num_anchors, -1)
    mu_p = anchor_mu.unsqueeze(0).expand(batch_size, -1, -1)
    var_p = anchor_var.unsqueeze(0).expand(batch_size, -1, -1)
    
    neg_kl = -0.5 * (
        torch.log(var_p + 1e-8) - log_var.unsqueeze(1) +
        (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1
    ).sum(dim=-1)  # (B, K)
    
    # Convert to probabilities (higher likelihood = higher probability)
    soft_labels = F.softmax(neg_kl, dim=-1).transpose(0, 1)  # (K, B)
    
    return soft_labels