"""
Losses and metrics.

huber_loss          : standard Huber, delta=1.0 calibrated for metre-scale absolute positions.
weighted_huber_loss : same but with a linear per-timestep weight ramp (near→far).
                      Directly addresses ade_far collapse by giving far-horizon
                      timesteps stronger gradient signal.
"""

import torch
import torch.nn.functional as F


def huber_loss(pred, target, delta=1.0):
    return F.huber_loss(pred, target[..., :2], delta=delta, reduction="mean")


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    near_weight: float = 0.5,
    far_weight: float = 3.0,
) -> torch.Tensor:
    """
    Huber loss with a linear per-timestep weight ramp from near_weight → far_weight.

    Motivation: ade_early ≈ 0.25, ade_far ≈ 4.0. Uniform loss means the
    near-horizon steps (easy, small error) dominate the gradient, starving
    the model of signal to improve far-horizon. Ramping the weight forces the
    optimiser to prioritise long-horizon accuracy.

    Default ramp 0.5 → 3.0 gives far-horizon steps 6× the gradient weight
    of near-horizon steps without completely ignoring them.
    """
    T = pred.size(1)
    weights = torch.linspace(near_weight, far_weight, T, device=pred.device)  # (T,)
    err = F.huber_loss(pred, target[..., :2], delta=delta, reduction="none")  # (B,T,2)
    err = err.mean(dim=-1)  # (B,T)
    return (err * weights.unsqueeze(0)).mean()


def get_loss_fn(cfg: dict):
    """Return the loss function specified in config."""
    name = cfg.get("loss", "huber")
    near_weight = cfg.get("loss_near_weight", 0.5)
    far_weight = cfg.get("loss_far_weight", 3.0)
    delta = cfg.get("loss_delta", 1.0)
    if name == "weighted_huber":
        return lambda pred, target: weighted_huber_loss(
            pred,
            target,
            delta=delta,
            near_weight=near_weight,
            far_weight=far_weight,
        )
    return lambda pred, target: huber_loss(pred, target, delta=delta)


@torch.no_grad()
def ade(pred, target):
    return torch.norm(pred - target[..., :2], p=2, dim=-1).mean()


@torch.no_grad()
def fde(pred, target):
    return torch.norm(pred[:, -1] - target[:, -1, :2], p=2, dim=-1).mean()
