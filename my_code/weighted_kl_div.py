import torch
import torch.nn.functional as F

def _compute_kl_matrix(mu, log_var, anchor_mu, anchor_var):
    """
    Compute KL(q||p_k) for each sample/anchor pair.

    Returns:
        kl_div: (B, K)
    """
    batch_size = mu.size(0)
    num_anchors = anchor_mu.size(0)

    if num_anchors == 0:
        return mu.new_zeros((batch_size, 0))

    # Expand dimensions for broadcasting
    mu_q = mu.unsqueeze(1).expand(-1, num_anchors, -1)  # (B, K, D)
    log_var_q = log_var.unsqueeze(1).expand(-1, num_anchors, -1)
    var_q = torch.exp(log_var_q)

    mu_p = anchor_mu.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, D)
    var_p = anchor_var.unsqueeze(0).expand(batch_size, -1, -1).clamp_min(1e-8)

    # KL(q||p) = 0.5 * [log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
    kl_div = 0.5 * (
        torch.log(var_p) - log_var_q +
        (var_q + (mu_q - mu_p) ** 2) / var_p - 1
    )
    return kl_div.sum(dim=-1)  # (B, K)


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
    num_anchors = anchor_mu.size(0)

    if num_anchors == 0:
        zero = mu.new_tensor(0.0)
        return zero, mu.new_zeros((0,))

    kl_div = _compute_kl_matrix(mu, log_var, anchor_mu, anchor_var)  # (B, K)
    kl_per_anchor = kl_div.mean(dim=0).detach()

    if soft_labels is not None:
        # Accept either (K, B) or (B, K)
        if soft_labels.shape == (num_anchors, mu.size(0)):
            weights = soft_labels.transpose(0, 1)  # (B, K)
        elif soft_labels.shape == (mu.size(0), num_anchors):
            weights = soft_labels
        else:
            raise ValueError(
                f"soft_labels shape {tuple(soft_labels.shape)} does not match "
                f"(K,B)=({num_anchors},{mu.size(0)}) or (B,K)=({mu.size(0)},{num_anchors})."
            )

        # If not already normalized probs, convert to probs.
        row_sums = weights.sum(dim=-1, keepdim=True)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            if temperature != 1.0:
                weights = F.softmax(weights / temperature, dim=-1)
            else:
                weights = F.softmax(weights, dim=-1)
        else:
            weights = weights / row_sums.clamp_min(1e-8)

        weighted_kl = torch.sum(kl_div * weights, dim=-1)  # (B,)
        kl_loss = weighted_kl.mean()
    else:
        # No soft labels: average KL across anchors and batch.
        kl_loss = kl_div.mean()

    return kl_loss, kl_per_anchor


def compute_soft_labels(mu, log_var, anchor_mu, anchor_var, temperature=1.0):
    """
    Compute soft assignment probabilities based on likelihood of z belonging to each anchor.
    
    Returns:
        soft_labels: (num_anchors, batch_size) - probability each sample belongs to each anchor
    """
    num_anchors = anchor_mu.size(0)

    if num_anchors == 0:
        return None

    kl_div = _compute_kl_matrix(mu, log_var, anchor_mu, anchor_var)  # (B, K)
    scaled_scores = -kl_div / max(temperature, 1e-8)
    soft_labels = F.softmax(scaled_scores, dim=-1).transpose(0, 1)  # (K, B)

    return soft_labels
