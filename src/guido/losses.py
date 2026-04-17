"""
Loss functions and evaluation metrics for trajectory prediction.

Training loss : Huber (smooth-L1) over predicted vs. ground-truth (x, y).
               More robust to outlier frames than plain MSE, which can blow up
               on the occasional wildly mis-predicted waypoint early in training.

Eval metrics  : ADE and FDE, matching the competition definition exactly.
"""

import torch
import torch.nn.functional as F


def huber_loss(
    pred: torch.Tensor,  # (B, T, 2)  predicted (x, y)
    target: torch.Tensor,  # (B, T, ≥2) ground-truth; only first 2 dims used
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber / smooth-L1 loss on the (x, y) dimensions."""
    return F.huber_loss(pred, target[..., :2], delta=delta, reduction="mean")


@torch.no_grad()
def ade(
    pred: torch.Tensor,  # (B, T, 2)
    target: torch.Tensor,  # (B, T, ≥2)
) -> torch.Tensor:
    """
    Average Displacement Error – mean per-step L2 distance, averaged over
    all timesteps and all samples in the batch.
    """
    return torch.norm(pred - target[..., :2], p=2, dim=-1).mean()


@torch.no_grad()
def fde(
    pred: torch.Tensor,  # (B, T, 2)
    target: torch.Tensor,  # (B, T, ≥2)
) -> torch.Tensor:
    """
    Final Displacement Error – L2 distance at the last predicted timestep,
    averaged over all samples in the batch.
    """
    return torch.norm(pred[:, -1] - target[:, -1, :2], p=2, dim=-1).mean()
