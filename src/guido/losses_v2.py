"""
Losses for DrivingPlannerV2 coarse-to-fine.
"""

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _ade_per_head(preds: list, target: torch.Tensor) -> torch.Tensor:
    """(K, B) ADE, no gradient — used only for WTA winner selection."""
    target_xy = target[..., :2]
    with torch.no_grad():
        return torch.stack(
            [torch.norm(p.detach() - target_xy, p=2, dim=-1).mean(dim=-1) for p in preds], dim=0
        )


def smoothness_loss(traj: torch.Tensor) -> torch.Tensor:
    vel = traj[:, 1:] - traj[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return (acc**2).sum(dim=-1).mean()


def weighted_huber(pred, target, delta=1.0, near_weight=1.0, far_weight=1.0):
    T = pred.size(1)
    weights = torch.linspace(near_weight, far_weight, T, device=pred.device)
    err = F.huber_loss(pred, target[..., :2], delta=delta, reduction="none")
    return (err.mean(dim=-1) * weights.unsqueeze(0)).mean()


def coarse_loss(coarse_preds: list, target: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary loss on coarse anchor waypoints.
    Supervise each coarse head at uniformly-spaced anchor indices of the GT.
    e.g. n_coarse=6 → GT indices [9, 19, 29, 39, 49, 59]
    This teaches the coarse heads to produce meaningful trajectory sketches
    before the router starts comparing them.
    """
    T = target.size(1)
    n = coarse_preds[0].size(1)
    # Uniformly spaced indices: last one is always step T-1
    indices = torch.linspace(T / n - 1, T - 1, n, device=target.device).long()
    gt_anchors = target[:, indices, :2]  # (B, n, 2)
    loss = sum(F.mse_loss(c, gt_anchors) for c in coarse_preds) / len(coarse_preds)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Main training loss
# ─────────────────────────────────────────────────────────────────────────────


def winner_takes_all_loss(
    fine_preds: list,
    coarse_preds: list,
    router_logits: torch.Tensor,
    target: torch.Tensor,
    smoothness_weight: float = 0.05,
    coarse_weight: float = 0.3,  # weight for coarse auxiliary loss
    router_weight: float = 0.5,
    wta_weight: float = 0.8,
    delta: float = 1.0,
    near_weight: float = 0.5,
    far_weight: float = 2.0,
    router_active: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    K = len(fine_preds)
    B = fine_preds[0].size(0)

    # WTA selection on fine predictions
    ade_kb = _ade_per_head(fine_preds, target)  # (K, B), no grad
    winner = ade_kb.argmin(dim=0)  # (B,)

    stacked = torch.stack(fine_preds, dim=0)
    winner_exp = winner.view(1, B, 1, 1).expand(1, B, stacked.size(2), 2)
    best_preds = stacked.gather(0, winner_exp).squeeze(0)  # (B, T, 2)

    # Fine trajectory loss (WTA + shared)
    wta_l = weighted_huber(best_preds, target, delta, near_weight, far_weight)
    shared_l = sum(weighted_huber(p, target, delta, near_weight, far_weight) for p in fine_preds) / K
    smooth_l = smoothness_loss(best_preds)

    # Coarse auxiliary loss (all heads, always on)
    # This ensures coarse heads produce meaningful sketches for the router
    coarse_l = coarse_loss(coarse_preds, target)

    loss = (
        wta_weight * wta_l
        + (1.0 - wta_weight) * shared_l
        + smoothness_weight * smooth_l
        + coarse_weight * coarse_l
    )

    # Router CE: only after warmup, only K > 1
    if K > 1 and router_active:
        router_l = F.cross_entropy(router_logits, winner)
        loss = loss + router_weight * router_l

    return loss, winner


# ─────────────────────────────────────────────────────────────────────────────
# Eval metrics
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def ade(pred, target):
    return torch.norm(pred - target[..., :2], p=2, dim=-1).mean()


@torch.no_grad()
def fde(pred, target):
    return torch.norm(pred[:, -1] - target[:, -1, :2], p=2, dim=-1).mean()


@torch.no_grad()
def best_of_k_ade(preds, target):
    return _ade_per_head(preds, target).min(dim=0).values.mean()
