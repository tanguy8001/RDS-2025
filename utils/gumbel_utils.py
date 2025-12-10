"""
Gumbel-Sparsemax utilities for CL-LoRA task selection.

Implements:
- Gumbel noise sampling
- Sparsemax projection (yields exact zeros, unlike softmax)
- Gumbel-Sparsemax forward pass
- Sparsity regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, device='cuda', eps=1e-20):
    # Sample from Gumbel(0, 1) distribution: −log(−log(Uniform(0,1)))
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def sparsemax(logits, dim=-1):
    """
    Sparsemax activation function (Martins & Astudillo, 2016).

    Projects logits onto the probability simplex, yielding sparse probabilities
    with exact zeros.

    Args:
        logits: Input logits, shape [..., num_classes]
        dim: Dimension to apply sparsemax over

    Returns:
        Sparse probability distribution with exact zeros

    Reference:
        "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
        https://arxiv.org/abs/1602.02068
    """
    # Replace -inf with very large negative number for numerical stability
    logits = torch.where(torch.isinf(logits), torch.full_like(logits, -1e9), logits)

    # Sort logits in descending order
    logits_sorted, _ = torch.sort(logits, dim=dim, descending=True)

    # Compute cumulative sum
    cumsum = torch.cumsum(logits_sorted, dim=dim)

    # Compute k(z) - the number of non-zero elements
    # k(z) = max{k : 1 + k * z_k > sum_{j=1}^{k} z_j}
    arange = torch.arange(1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype)

    # Expand arange to match logits shape
    shape = [1] * len(logits.shape)
    shape[dim] = -1
    arange = arange.view(*shape)

    threshold = (cumsum - 1) / arange

    # Find support: where sorted logits > threshold
    support = (logits_sorted > threshold).float()

    # Compute k: number of elements in support
    k = support.sum(dim=dim, keepdim=True)

    # Compute tau(z): the threshold value
    tau_sum = (logits_sorted * support).sum(dim=dim, keepdim=True)
    tau = (tau_sum - 1) / (k + 1e-8)

    # Apply sparsemax transformation
    output = torch.clamp(logits - tau, min=0.0)

    # Normalize to ensure probabilities sum to 1
    output_sum = output.sum(dim=dim, keepdim=True)
    output = output / (output_sum + 1e-8)

    return output


def gumbel_sparsemax(logits, tau=1.0, hard=False, dim=-1, training=True):
    """
    Gumbel-Sparsemax: sparse differentiable sampling.

    Combines Gumbel noise with Sparsemax projection for sparse task selection.
    During training, uses soft selection. During evaluation or with hard=True,
    uses hard selection (one-hot or top-k).

    Args:
        logits: Gate logits, shape [..., num_tasks]
        tau: Temperature parameter (higher = more exploration)
        hard: If True, return hard (one-hot) selection
        dim: Dimension to apply sparsemax over
        training: Whether in training mode

    Returns:
        beta: Task selection weights (sparse, sums to 1)
    """
    if training:
        gumbel_noise = sample_gumbel(logits.shape, device=logits.device)

        noisy_logits = (logits + gumbel_noise) / tau
    else:
        # No noise during eval
        noisy_logits = logits / tau

    soft_beta = sparsemax(noisy_logits, dim=dim)

    if hard:
        # Hard selection: keep only max val
        # Create one-hot vector for backprop
        hard_beta = torch.zeros_like(soft_beta)
        max_indices = torch.argmax(soft_beta, dim=dim, keepdim=True)
        hard_beta.scatter_(dim, max_indices, 1.0)

        # Straight-through estimator: forward = hard, backward = soft
        beta = hard_beta - soft_beta.detach() + soft_beta
    else:
        beta = soft_beta

    return beta


def sparsity_loss(beta, eps=1e-8):
    """
    Entropy-based sparsity regularization (Eq. 3 in paper).
    Loss = -Σ β_i log(β_i + ε)
    """
    # Beta is (bs, num_tasks) or (num_tasks)
    # Negative entropy (encourages low entropy = high sparsity)
    entropy = -torch.sum(beta * torch.log(beta + eps), dim=-1)
    return entropy.mean()


class TemperatureScheduler:
    """
    Temperature annealing scheduler for Gumbel-Sparsemax.

    Implements exponential decay: τ_t = τ_final + (τ_init - τ_final) * decay^t

    Args:
        tau_init: Initial temperature (high = more exploration)
        tau_final: Final temperature (low = more exploitation)
        anneal_rate: Decay rate per step (e.g., 0.999)

    Example:
        >>> scheduler = TemperatureScheduler(tau_init=5.0, tau_final=0.5, anneal_rate=0.999)
        >>> for epoch in range(100):
        ...     tau = scheduler.step()
        ...     # Use tau in gumbel_sparsemax
    """
    def __init__(self, tau_init=5.0, tau_final=0.5, anneal_rate=0.999):
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.anneal_rate = anneal_rate
        self.current_step = 0

    def step(self):
        """Get current temperature and increment step counter."""
        tau = self.tau_final + (self.tau_init - self.tau_final) * (self.anneal_rate ** self.current_step)
        self.current_step += 1
        return max(tau, self.tau_final)  # Ensure we don't go below tau_final

    def reset(self):
        self.current_step = 0

    def get_temperature(self):
        """Get current temperature without incrementing."""
        return self.tau_final + (self.tau_init - self.tau_final) * (self.anneal_rate ** self.current_step)


def hard_selection(beta, threshold=0.1):
    return (beta > threshold).float()
