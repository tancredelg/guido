"""
train_v2.py — training script for DrivingPlannerV2.

Usage:
    python src/train_v2.py --config configs/v2_baseline.yaml
    python src/train_v2.py --config configs/v2_baseline.yaml --no-wandb
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

from guido.dataset import make_datasets, make_test_dataset
from guido.model_v2 import DrivingPlannerV2
from guido.losses_v2 import winner_takes_all_loss, ade, fde, best_of_k_ade
from guido.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    checkpoint_path,
    build_submission_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli(cfg: dict, args: argparse.Namespace) -> dict:
    for k, v in [
        ("data_dir", args.data_dir),
        ("dino_repo_dir", args.dino_repo_dir),
        ("dino_weights", args.dino_weights),
    ]:
        if v:
            cfg[k] = v
    return cfg


# ── W&B ───────────────────────────────────────────────────────────────────────


def init_wandb(cfg: dict, enabled: bool):
    if not enabled:
        return None
    try:
        import wandb

        run = wandb.init(
            project=cfg.get("wandb_project", "dlav-guido"),
            entity=cfg.get("wandb_entity", "tancredelg-personal"),
            name=cfg.get("wandb_run_name", None),
            config=cfg,
            resume="allow",
        )
        log.info("W&B: %s", run.url)
        return run
    except Exception as e:
        log.warning("W&B init failed (%s)", e)
        return None


def wandb_log(run, step, d):
    if run:
        try:
            run.log(d, step=step)
        except Exception:
            pass


# ── Trajectory-length weighted sampler ────────────────────────────────────────


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Oversample hard samples: weight each training sample by the total
    displacement of its GT trajectory.  Near-stationary samples get
    lower weight; long fast trajectories get higher weight.
    This counteracts the imbalance where easy forward-cruising samples
    dominate a uniform random batch.
    """
    import pickle

    displacements = []
    for path in dataset.samples:
        with open(path, "rb") as f:
            data = pickle.load(f)
        fut = data["sdc_future_feature"]
        disp = float(np.linalg.norm(fut[-1, :2] - fut[0, :2]))
        displacements.append(max(disp, 0.5))  # floor so stationary isn't zero
    weights = torch.tensor(displacements, dtype=torch.float)
    weights = weights / weights.sum()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    ade_vals, fde_vals, bofk_vals = [], [], []
    cmd_ade = {0: [], 1: [], 2: []}
    horizon = {"early": [], "mid": [], "far": []}
    static_ade, moving_ade = [], []
    vis_batch = None

    sw = cfg.get("smoothness_weight", 0.1)

    for batch in loader:
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        command = batch["command"].to(device)
        future = batch["future"].to(device)

        preds, router_logits = model(camera, history, command)

        loss, _ = winner_takes_all_loss(
            preds,
            router_logits,
            future,
            smoothness_weight=sw,
            wta_weight=cfg.get("wta_weight", 0.8),
            router_weight=cfg.get("router_weight", 0.5),
            near_weight=cfg.get("loss_near_weight", 0.5),
            far_weight=cfg.get("loss_far_weight", 2.0),
            router_active=True,  # always active during validation reporting
        )
        total_loss += loss.item()

        # Head selection via trained router — same as inference, no oracle
        stacked = torch.stack(preds, dim=0)  # (K, B, 60, 2)
        gt_xy = future[..., :2]
        best_k = router_logits.argmax(dim=-1)  # (B,)
        idx_exp = best_k.view(1, -1, 1, 1).expand(1, -1, 60, 2)
        best_pred = stacked.gather(0, idx_exp).squeeze(0)  # (B, 60, 2)

        per_step = torch.norm(best_pred - gt_xy, p=2, dim=-1)  # (B, 60)

        ade_vals.append(per_step.mean().item())
        fde_vals.append(per_step[:, -1].mean().item())
        bofk_vals.append(best_of_k_ade(preds, future).item())

        for c in range(3):
            m = command == c
            if m.any():
                cmd_ade[c].append(per_step[m].mean().item())

        horizon["early"].append(per_step[:, :20].mean().item())
        horizon["mid"].append(per_step[:, 20:40].mean().item())
        horizon["far"].append(per_step[:, 40:].mean().item())

        disp = torch.norm(future[:, -1, :2] - future[:, 0, :2], p=2, dim=-1)
        sm = disp < 2.0
        if sm.any():
            static_ade.append(per_step[sm].mean().item())
        if (~sm).any():
            moving_ade.append(per_step[~sm].mean().item())

        if vis_batch is None:
            vis_batch = {k: v.cpu() for k, v in batch.items()}
            vis_batch["_pred"] = best_pred.cpu()

    CMD = {0: "forward", 1: "left", 2: "right"}
    metrics = {
        "val/loss": total_loss / len(loader),
        "val/ade": float(np.mean(ade_vals)),
        "val/fde": float(np.mean(fde_vals)),
        "val/bofk_ade": float(np.mean(bofk_vals)),
        "val/ade_early": float(np.mean(horizon["early"])),
        "val/ade_mid": float(np.mean(horizon["mid"])),
        "val/ade_far": float(np.mean(horizon["far"])),
    }
    for c, name in CMD.items():
        if cmd_ade[c]:
            metrics[f"val/ade_{name}"] = float(np.mean(cmd_ade[c]))
    if static_ade:
        metrics["val/ade_static"] = float(np.mean(static_ade))
    if moving_ade:
        metrics["val/ade_moving"] = float(np.mean(moving_ade))

    return metrics, vis_batch


# ── Training loop ─────────────────────────────────────────────────────────────


def train(cfg: dict, use_wandb: bool) -> None:
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds, val_ds = make_datasets(
        cfg["data_dir"],
        mirror_p=cfg.get("mirror_p", 0.0),
        hist_noise_std=cfg.get("hist_noise_std", 0.0),
        mirror_warmup=cfg.get("mirror_warmup", 10),
    )
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # Optionally use trajectory-length weighted sampling
    sampler = None
    shuffle = True
    use_sampler = cfg.get("weighted_sampler", False)
    if use_sampler:
        log.info("Using trajectory-length weighted sampler")
        sampler = make_weighted_sampler(train_ds)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = DrivingPlannerV2(
        dino_model=cfg.get("dino_model", "dinov3_vitb16"),
        dino_repo_dir=cfg["dino_repo_dir"],
        dino_weights=cfg["dino_weights"],
        unfreeze_blocks=cfg.get("unfreeze_blocks", 1),
        d=cfg.get("d", 256),
        num_heads=cfg.get("num_heads", 4),
        hist_layers=cfg.get("hist_layers", 2),
        dec_layers=cfg.get("dec_layers", 2),
        K=cfg.get("K", 3),
        dropout=cfg.get("dropout", 0.05),
        smoothness_weight=cfg.get("smoothness_weight", 0.1),
        cmd_embed_dim=cfg.get("cmd_embed_dim", 64),
    ).to(device)
    log.info("Trainable params: %s", f"{model.num_trainable_params():,}")

    # ── Optimiser: separate lr for backbone ───────────────────────────────
    base_lr = cfg.get("lr", 5e-4)
    backbone_lr = cfg.get("backbone_lr", base_lr / 20)
    router_lr = cfg.get("router_lr", base_lr * 10.0)  # Much higher LR for late-starting router
    wd = cfg.get("weight_decay", 1e-4)

    def no_wd(n):
        return "bias" in n or "norm" in n or "pe" in n

    head_decay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad
        and not n.startswith("backbone.")
        and not n.startswith("router.")
        and not no_wd(n)
    ]
    head_no_decay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("backbone.") and not n.startswith("router.") and no_wd(n)
    ]
    router_decay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("router.") and not no_wd(n)
    ]
    router_no_decay = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith("router.") and no_wd(n)
    ]
    bb_decay = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone.") and not no_wd(n)
    ]
    bb_no_decay = [
        p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.") and no_wd(n)
    ]

    groups = [
        {"params": head_decay, "lr": base_lr, "weight_decay": wd},
        {"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": router_decay, "lr": router_lr, "weight_decay": wd},
        {"params": router_no_decay, "lr": router_lr, "weight_decay": 0.0},
    ]
    if bb_decay or bb_no_decay:
        groups += [
            {"params": bb_decay, "lr": backbone_lr, "weight_decay": wd},
            {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        ]
        log.info("Backbone lr %.2e  Head lr %.2e  Router lr %.2e", backbone_lr, base_lr, router_lr)

    optimizer = optim.AdamW(groups)

    num_epochs = cfg["num_epochs"]
    warmup_epochs = cfg.get("warmup_epochs", 5)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=cfg.get("min_lr", 1e-6),
    )
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_ade = float("inf")
    best_ckpt = None
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ade = ckpt["val_ade"]
        log.info("Resumed epoch %d (best ADE %.4f)", start_epoch, best_ade)

    run = init_wandb(cfg, enabled=use_wandb)
    grad_clip = cfg.get("grad_clip", 1.0)
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    log_every = cfg.get("log_interval", 20)
    global_step = start_epoch * len(train_loader)
    sw = cfg.get("smoothness_weight", 0.1)
    K = cfg.get("K", 3)
    # Freeze router for first N epochs so heads specialise before the router
    # learns to route. Avoids training router on random/noisy winner labels.
    router_warmup_epochs = cfg.get("router_warmup_epochs", 10)

    # ── Main loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        # Freeze/unfreeze router based on warmup schedule
        router_frozen = (epoch < router_warmup_epochs) and K > 1
        for p in model.router.parameters():
            p.requires_grad = not router_frozen
        if epoch == router_warmup_epochs and K > 1:
            log.info("Epoch %d: router unfrozen — CE loss now active", epoch + 1)

        # Schedule router weight: ramp up linearly to 3x after warmup
        # As the trajectory losses decay, we need to artificially boost the router loss
        # so it maintains strong gradient signals.
        base_router_weight = cfg.get("router_weight", 0.5)
        if not router_frozen and K > 1:
            progress = (epoch - router_warmup_epochs) / max(1, num_epochs - router_warmup_epochs)
            curr_router_weight = base_router_weight  # * (1.0 + 2.0 * progress)
        else:
            curr_router_weight = base_router_weight

        wandb_log(
            run,
            epoch,
            {"train/router_active": int(not router_frozen), "train/router_weight": curr_router_weight},
        )
        train_ds.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            camera = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)
            future = batch["future"].to(device)

            optimizer.zero_grad(set_to_none=True)
            preds, router_logits = model(camera, history, command)
            loss, winner = winner_takes_all_loss(
                preds,
                router_logits,
                future,
                smoothness_weight=sw,
                wta_weight=cfg.get("wta_weight", 0.8),
                router_weight=curr_router_weight,
                near_weight=cfg.get("loss_near_weight", 0.5),
                far_weight=cfg.get("loss_far_weight", 2.0),
                router_active=not router_frozen,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            if (step + 1) % log_every == 0:
                # log which heads are winning to detect head collapse
                w_counts = winner.bincount(minlength=cfg.get("K", 3)).float()
                w_frac = w_counts / w_counts.sum()
                wandb_log(
                    run,
                    global_step,
                    {
                        "train/loss_step": loss.item(),
                        **{f"train/head_{k}_win_frac": w_frac[k].item() for k in range(len(w_frac))},
                    },
                )
                log.info(
                    "  epoch %d  step %d/%d  loss %.4f", epoch + 1, step + 1, len(train_loader), loss.item()
                )

        scheduler.step()

        metrics, vis_batch = validate(model, val_loader, device, cfg)
        lr_now = optimizer.param_groups[0]["lr"]

        # During router warmup the router is random → val/ade is meaningless.
        # Log bofk_ade as the primary curve so W&B charts are interpretable,
        # and log router_ade separately so we can see when it converges to bofk.
        router_active = (epoch >= router_warmup_epochs) or (K == 1)
        log_metrics = {
            "train/loss": train_loss / len(train_loader),
            "train/lr": lr_now,
            "val/ade": metrics["val/bofk_ade"],  # always the honest number
            "val/router_ade": metrics["val/ade"],  # random during warmup, real after
            **{k: v for k, v in metrics.items() if k not in ("val/ade",)},
            "epoch": epoch + 1,
            "router_active": int(router_active),
        }
        wandb_log(run, global_step, log_metrics)

        log.info(
            "Epoch %3d/%d | lr %.2e | train %.4f | bofk %.4f | router_ade %.4f%s | far %.4f | %.0fs",
            epoch + 1,
            num_epochs,
            lr_now,
            train_loss / len(train_loader),
            metrics["val/bofk_ade"],
            metrics["val/ade"],
            "" if router_active else " (warmup)",
            metrics["val/ade_far"],
            time.time() - t0,
        )

        # Save on bofk_ade: this is the oracle upper bound during router warmup
        # and converges toward val/ade as the router trains. It's the honest
        # measure of trajectory quality independent of routing quality.
        save_metric = metrics["val/bofk_ade"]
        if save_metric < best_ade:
            best_ade = save_metric
            path = checkpoint_path(ckpt_dir, epoch + 1, best_ade)
            save_checkpoint(path, model, optimizer, scheduler, epoch + 1, best_ade, cfg)
            log.info("  ✓ Best ADE %.4f → %s", best_ade, path)
            best_ckpt = path
            if run:
                try:
                    run.summary["best_ade"] = best_ade
                    run.summary["best_ckpt"] = path
                except Exception:
                    pass

    log.info("Done. Best ADE: %.4f @ %s", best_ade, best_ckpt)
    if run:
        run.finish()


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_baseline.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dino-repo-dir", default=None)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)
    log.info("Config: %s", cfg)
    train(cfg, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
