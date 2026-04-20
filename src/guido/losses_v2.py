"""
Losses for DrivingPlannerV2.
"""

import torch
import torch.nn.functional as F


def _ade_per_head(preds: list, target: torch.Tensor) -> torch.Tensor:
    """
    preds  : list of K tensors, each (B, T, 2)
    target : (B, T, ≥2)
    Returns: (K, B) ADE per head per sample — no gradient flows through this.
    """
    target_xy = target[..., :2].detach()
    with torch.no_grad():
        ades = []
        for p in preds:
            err = torch.norm(p.detach() - target_xy, p=2, dim=-1).mean(dim=-1)
            ades.append(err)
    return torch.stack(ades, dim=0)  # (K, B)


def smoothness_loss(traj: torch.Tensor) -> torch.Tensor:
    vel = traj[:, 1:] - traj[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    return (acc**2).sum(dim=-1).mean()


def weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    near_weight: float = 1.0,
    far_weight: float = 1.0,
) -> torch.Tensor:
    T = pred.size(1)
    weights = torch.linspace(near_weight, far_weight, T, device=pred.device)
    err = F.huber_loss(pred, target[..., :2], delta=delta, reduction="none")
    err = err.mean(dim=-1)  # (B, T)
    return (err * weights.unsqueeze(0)).mean()


def winner_takes_all_loss(
    preds: list,
    router_logits: torch.Tensor,
    target: torch.Tensor,
    smoothness_weight: float = 0.05,
    router_weight: float = 0.5,
    wta_weight: float = 0.8,
    delta: float = 1.0,
    near_weight: float = 0.5,
    far_weight: float = 2.0,
    router_active: bool = True,  # False during router warmup freeze
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft WTA trajectory loss + optional router cross-entropy.

    router_active must be False during the router warmup period.
    When False, the router CE term is completely excluded from the loss —
    no gradient flows from it to the encoders.  This is critical: even
    with router.requires_grad=False, F.cross_entropy still backprops
    through router_logits into the encoder params that produced them.
    """
    K = len(preds)
    B = preds[0].size(0)

    # Pick winner per sample — done without gradient to avoid biasing heads
    ade_kb = _ade_per_head(preds, target)  # (K, B), no grad
    winner = ade_kb.argmin(dim=0)  # (B,)

    # Gather the winning head's predictions WITH gradient for backprop
    stacked = torch.stack(preds, dim=0)  # (K, B, T, 2)
    winner_exp = winner.view(1, B, 1, 1).expand(1, B, stacked.size(2), 2)
    best_preds = stacked.gather(0, winner_exp).squeeze(0)  # (B, T, 2)

    # WTA component: winning head only
    wta_loss = weighted_huber(best_preds, target, delta, near_weight, far_weight)

    # Shared component: all heads (prevents head starvation / collapse)
    shared_loss = sum(weighted_huber(p, target, delta, near_weight, far_weight) for p in preds) / K

    smooth_loss = smoothness_loss(best_preds)

    loss = wta_weight * wta_loss + (1.0 - wta_weight) * shared_loss + smoothness_weight * smooth_loss

    # Router CE: only when K > 1 AND router is past its warmup freeze.
    # IMPORTANT: when router_active=False we skip entirely — not just the
    # router weight update, but the whole term, so zero gradient flows
    # backwards from a random untrained router into the encoders.
    if K > 1 and router_active:
        router_loss = F.cross_entropy(router_logits, winner)
        loss = loss + router_weight * router_loss

    return loss, winner


@torch.no_grad()
def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target[..., :2], p=2, dim=-1).mean()


@torch.no_grad()
def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[:, -1] - target[:, -1, :2], p=2, dim=-1).mean()


@torch.no_grad()
def best_of_k_ade(preds: list, target: torch.Tensor) -> torch.Tensor:
    ade_kb = _ade_per_head(preds, target)
    return ade_kb.min(dim=0).values.mean()
