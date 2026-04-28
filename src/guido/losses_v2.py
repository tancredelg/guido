"""
Losses for DrivingPlannerV2 coarse-to-fine.
"""

import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ade_per_head(preds: list, target: torch.Tensor) -> torch.Tensor:
    """(K, B) ADE, no gradient — used only for WTA winner selection."""
    target_xy = target[..., :2]
    with torch.no_grad():
        return torch.stack([
            torch.norm(p.detach() - target_xy, p=2, dim=-1).mean(dim=-1)
            for p in preds
        ], dim=0)


def smoothness_loss(traj: torch.Tensor) -> torch.Tensor:
    vel = traj[:, 1:] - traj[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return (acc ** 2).sum(dim=-1).mean()


def weighted_huber(pred, target, delta=1.0, near_weight=1.0, far_weight=1.0,
                   weighting: str = "linear"):
    """
    Per-step weighted Huber loss.

    weighting: 'linear'  → near_weight → far_weight ramp (default)
               'u'       → U-shape: high at both endpoints, lower in the middle.
                            Anchors trajectory at start AND end.  Useful when
                            you want strong constraints on both ADE_start and FDE.
                            Min weight = mid-trajectory; max weights at t=0 and t=T-1.
    """
    T = pred.size(1)
    device = pred.device
    if weighting == "u":
        # U-shape: weight = max(near, far) - (max-min) * sin(pi*t/(T-1))
        # Concretely: w(t=0) = near_w, w(t=T-1) = far_w, w(mid) ~ min(near, far) * 0.5
        t      = torch.linspace(0, 1, T, device=device)
        # Scale endpoints up, dip in the middle
        u_min  = min(near_weight, far_weight) * 0.5
        bell   = torch.sin(t * math.pi)              # 0 at endpoints, 1 mid
        weights = torch.where(
            t < 0.5,
            near_weight - (near_weight - u_min) * bell,
            far_weight  - (far_weight  - u_min) * bell,
        )
    else:
        weights = torch.linspace(near_weight, far_weight, T, device=device)

    err = F.huber_loss(pred, target[..., :2], delta=delta, reduction="none")
    return (err.mean(dim=-1) * weights.unsqueeze(0)).mean()


def fde_loss(pred, target, delta=1.0):
    """
    Direct supervision on the final waypoint — Huber distance between
    predicted and GT final position. Hits FDE metric directly and provides
    a strong endpoint anchor that prevents far-horizon drift.
    """
    return F.huber_loss(pred[:, -1], target[:, -1, :2], delta=delta, reduction="mean")


def stationary_penalty(pred, target, threshold: float = 2.0):
    """
    Penalty on velocity magnitude for samples whose GT trajectory is short
    (total displacement < threshold meters). Encourages the model to predict
    near-zero motion when the car is stationary, fixing the jittery
    micro-movements seen in stopped scenarios.

    Returns 0 if no stationary samples in batch.
    """
    # Total GT displacement per sample
    total_disp = torch.norm(target[:, -1, :2] - target[:, 0, :2], p=2, dim=-1)  # (B,)
    static_mask = total_disp < threshold                                          # (B,)
    if not static_mask.any():
        return torch.tensor(0.0, device=pred.device)

    # Velocity = consecutive position differences
    vel = pred[:, 1:] - pred[:, :-1]                       # (B, T-1, 2)
    vel_mag = torch.norm(vel, p=2, dim=-1)                  # (B, T-1)
    # Penalise velocity magnitude only for stationary samples
    return vel_mag[static_mask].mean()


def coarse_loss(coarse_preds: list, target: torch.Tensor) -> torch.Tensor:
    """Auxiliary MSE on coarse anchor waypoints. Returns 0 when coarse_preds are None."""
    valid = [c for c in coarse_preds if c is not None]
    if not valid:
        return torch.tensor(0.0, device=target.device)
    T = target.size(1)
    n = valid[0].size(1)
    indices    = torch.linspace(T / n - 1, T - 1, n, device=target.device).long()
    gt_anchors = target[:, indices, :2]
    return sum(F.mse_loss(c, gt_anchors) for c in valid) / len(valid)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loss
# ─────────────────────────────────────────────────────────────────────────────

def winner_takes_all_loss(
    fine_preds: list,
    coarse_preds: list,
    router_logits: torch.Tensor,
    target: torch.Tensor,
    smoothness_weight: float  = 0.05,
    coarse_weight: float      = 0.3,
    router_weight: float      = 0.5,
    wta_weight: float         = 0.8,
    fde_weight: float         = 0.0,    # direct loss on final waypoint
    stationary_weight: float  = 0.0,    # velocity damping for static samples
    weighting: str            = "linear",
    delta: float              = 1.0,
    near_weight: float        = 0.5,
    far_weight: float         = 2.0,
    router_active: bool       = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    K = len(fine_preds)
    B = fine_preds[0].size(0)

    ade_kb = _ade_per_head(fine_preds, target)
    winner = ade_kb.argmin(dim=0)

    stacked    = torch.stack(fine_preds, dim=0)
    winner_exp = winner.view(1, B, 1, 1).expand(1, B, stacked.size(2), 2)
    best_preds = stacked.gather(0, winner_exp).squeeze(0)

    wta_l    = weighted_huber(best_preds, target, delta, near_weight, far_weight, weighting)
    shared_l = sum(
        weighted_huber(p, target, delta, near_weight, far_weight, weighting)
        for p in fine_preds
    ) / K
    smooth_l = smoothness_loss(best_preds)
    coarse_l = coarse_loss(coarse_preds, target)

    loss = (wta_weight * wta_l
            + (1.0 - wta_weight) * shared_l
            + smoothness_weight * smooth_l
            + coarse_weight * coarse_l)

    if fde_weight > 0:
        loss = loss + fde_weight * fde_loss(best_preds, target, delta)

    if stationary_weight > 0:
        loss = loss + stationary_weight * stationary_penalty(best_preds, target)

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