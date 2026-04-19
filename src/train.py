"""
train.py – Guido baseline training script.

Usage:
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/baseline.yaml --no-wandb
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from guido import (
    DrivingPlanner,
    make_datasets,
    get_loss_fn,
    ade as compute_ade,
    fde as compute_fde,
    seed_everything,
    save_checkpoint,
    checkpoint_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── W&B ───────────────────────────────────────────────────────────────────────


def init_wandb(cfg: dict, enabled: bool):
    if not enabled:
        return None
    try:
        import wandb

        run = wandb.init(
            project=cfg.get("wandb_project", "guido-dlav"),
            entity=cfg.get("wandb_entity", "tancredelg-personal"),
            name=cfg.get("wandb_run_name", None),
            config=cfg,
            resume="allow",
        )
        log.info("W&B run: %s", run.url)
        return run
    except Exception as e:
        log.warning("W&B init failed (%s) – continuing without it.", e)
        return None


def wandb_log(run, step: int, metrics: dict):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:
        pass


def wandb_log_trajectories(run, step: int, batch: dict, pred: torch.Tensor, n: int = 4):
    """Log predicted vs GT trajectory plots to W&B every N epochs."""
    if run is None:
        return
    try:
        import wandb
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])

        cameras = batch["camera"].cpu()
        history = batch["history"].cpu()
        future = batch["future"].cpu()
        pred_cpu = pred.detach().cpu()
        n = min(n, cameras.shape[0])

        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        for i in range(n):
            # Camera image
            img = cameras[i].permute(1, 2, 0).numpy() * STD + MEAN
            axes[0][i].imshow(np.clip(img, 0, 1))
            axes[0][i].axis("off")
            # Trajectory
            ax = axes[1][i]
            ax.plot(history[i, :, 0], history[i, :, 1], "o-", color="gold", ms=3, lw=1, label="history")
            ax.plot(future[i, :, 0], future[i, :, 1], "o-", color="limegreen", ms=3, lw=1, label="GT")
            ax.plot(pred_cpu[i, :, 0], pred_cpu[i, :, 1], "o-", color="tomato", ms=3, lw=1, label="pred")
            ax.set_aspect("equal")
            ax.legend(fontsize=7)
        plt.tight_layout()
        run.log({"val/trajectories": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        log.debug("W&B trajectory plot failed: %s", e)


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli(cfg: dict, args: argparse.Namespace) -> dict:
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.dino_repo_dir:
        cfg["dino_repo_dir"] = args.dino_repo_dir
    if args.dino_weights:
        cfg["dino_weights"] = args.dino_weights
    if args.epochs:
        cfg["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    return cfg


# ── Validation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    ade_vals, fde_vals = [], []
    # Per-command: 0=forward, 1=left, 2=right
    cmd_ade = {0: [], 1: [], 2: []}
    # Per-horizon buckets: early(0-19), mid(20-39), far(40-59)
    horizon_ade = {"early": [], "mid": [], "far": []}
    # Stationary vs moving (threshold: total GT displacement < 2m)
    static_ade, moving_ade = [], []
    vis_batch = None

    for batch in loader:
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        command = batch["command"].to(device)
        future = batch["future"].to(device)  # (B, 60, 3)

        pred = model(camera, history, command)
        total_loss += loss_fn(pred, future).item()

        per_step_err = torch.norm(pred - future[..., :2], p=2, dim=-1)  # (B, 60)
        ade_vals.append(per_step_err.mean().item())
        fde_vals.append(per_step_err[:, -1].mean().item())

        # Per-command ADE
        for c in range(3):
            mask = command == c
            if mask.any():
                cmd_ade[c].append(per_step_err[mask].mean().item())

        # Per-horizon ADE
        horizon_ade["early"].append(per_step_err[:, :20].mean().item())
        horizon_ade["mid"].append(per_step_err[:, 20:40].mean().item())
        horizon_ade["far"].append(per_step_err[:, 40:].mean().item())

        # Stationary vs moving: total displacement of GT trajectory
        total_disp = torch.norm(future[:, -1, :2] - future[:, 0, :2], p=2, dim=-1)  # (B,)
        static_mask = total_disp < 2.0
        if static_mask.any():
            static_ade.append(per_step_err[static_mask].mean().item())
        if (~static_mask).any():
            moving_ade.append(per_step_err[~static_mask].mean().item())

        if vis_batch is None:
            vis_batch = {k: v.cpu() for k, v in batch.items()}
            vis_batch["_pred"] = pred.cpu()

    CMD_NAMES = {0: "forward", 1: "left", 2: "right"}
    metrics = {
        "val/loss": total_loss / len(loader),
        "val/ade": float(np.mean(ade_vals)),
        "val/fde": float(np.mean(fde_vals)),
        "val/ade_early": float(np.mean(horizon_ade["early"])),
        "val/ade_mid": float(np.mean(horizon_ade["mid"])),
        "val/ade_far": float(np.mean(horizon_ade["far"])),
    }
    for c, name in CMD_NAMES.items():
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

    loss_fn = get_loss_fn(cfg)
    log.info(
        "Loss: %s  (near_w=%.1f far_w=%.1f)",
        cfg.get("loss", "huber"),
        cfg.get("loss_near_weight", 1.0),
        cfg.get("loss_far_weight", 1.0),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = make_datasets(
        cfg["data_dir"],
        mirror_p=cfg.get("mirror_p", 0.0),
        hist_noise_std=cfg.get("hist_noise_std", 0.0),
        mirror_warmup=cfg.get("mirror_warmup", 10),
    )
    log.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DrivingPlanner(
        dino_model=cfg.get("dino_model", "dinov3_vitb16"),
        dino_repo_dir=cfg["dino_repo_dir"],
        dino_weights=cfg["dino_weights"],
        hist_hidden_dim=cfg.get("hist_hidden_dim", 128),
        cmd_embed_dim=cfg.get("cmd_embed_dim", 32),
        fusion_dim=cfg.get("fusion_dim", 256),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.05),
        fusion_arch=cfg.get("fusion_arch", "concat"),
        decoder_arch=cfg.get("decoder_arch", "mlp"),
        decoder_d=cfg.get("decoder_d", 128),
        decoder_layers=cfg.get("decoder_layers", 2),
        n_anchors=cfg.get("n_anchors", 12),
        unfreeze_blocks=cfg.get("unfreeze_blocks", 0),
        residual_baseline=cfg.get("residual_baseline", False),
        decoder_patches=cfg.get("decoder_patches", False),
    ).to(device)
    log.info("Model ready – trainable params: %s", f"{model.num_trainable_params():,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # When unfreeze_blocks > 0, the unfrozen backbone blocks get backbone_lr
    # (typically 10-20× lower than the head lr) to avoid catastrophic forgetting.
    base_lr = cfg.get("lr", 5e-4)
    backbone_lr = cfg.get("backbone_lr", base_lr / 20)
    wd = cfg.get("weight_decay", 1e-4)

    def _split(params):
        decay = [p for n, p in params if "bias" not in n and "norm" not in n]
        no_decay = [p for n, p in params if "bias" in n or "norm" in n]
        return decay, no_decay

    head_named = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")
    ]
    backbone_named = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")
    ]

    hd, hnd = _split(head_named)
    bd, bnd = _split(backbone_named)

    param_groups = [
        {"params": hd, "lr": base_lr, "weight_decay": wd},
        {"params": hnd, "lr": base_lr, "weight_decay": 0.0},
    ]
    if bd or bnd:
        param_groups += [
            {"params": bd, "lr": backbone_lr, "weight_decay": wd},
            {"params": bnd, "lr": backbone_lr, "weight_decay": 0.0},
        ]
        log.info("Backbone fine-tune lr: %.2e  Head lr: %.2e", backbone_lr, base_lr)

    optimizer = optim.AdamW(param_groups)
    num_epochs = cfg["num_epochs"]
    warmup_epochs = cfg.get("warmup_epochs", 0)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=cfg.get("min_lr", 1e-6),
    )
    if warmup_epochs > 0:
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
    else:
        scheduler = cosine

    # ── Resume ────────────────────────────────────────────────────────────────
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

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
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
            pred = model(camera, history, command)
            loss = loss_fn(pred, future)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            global_step += 1
            if (step + 1) % log_every == 0:
                wandb_log(run, global_step, {"train/loss_step": loss.item()})
                log.info(
                    "  epoch %d  step %d/%d  loss %.4f", epoch + 1, step + 1, len(train_loader), loss.item()
                )

        scheduler.step()

        metrics, vis_batch = validate(model, val_loader, device, loss_fn)
        lr_now = scheduler.get_last_lr()[0]
        epoch_metrics = {
            "train/loss": train_loss / len(train_loader),
            "train/lr": lr_now,
            **metrics,
            "epoch": epoch + 1,
        }
        wandb_log(run, global_step, epoch_metrics)

        if (epoch + 1) % 5 == 0 and vis_batch is not None:
            wandb_log_trajectories(run, global_step, vis_batch, vis_batch["_pred"])

        log.info(
            "Epoch %3d/%d | lr %.2e | train_loss %.4f | val_loss %.4f | ADE %.4f | FDE %.4f | %.0fs",
            epoch + 1,
            num_epochs,
            lr_now,
            train_loss / len(train_loader),
            metrics["val/loss"],
            metrics["val/ade"],
            metrics["val/fde"],
            time.time() - t0,
        )

        if metrics["val/ade"] < best_ade:
            best_ade = metrics["val/ade"]
            path = checkpoint_path(ckpt_dir, epoch + 1, best_ade)
            save_checkpoint(path, model, optimizer, scheduler, epoch + 1, best_ade, cfg)
            log.info("  ✓ New best ADE %.4f → saved to %s", best_ade, path)
            best_ckpt = path
            if run:
                try:
                    run.summary["best_ade"] = best_ade
                    run.summary["best_ckpt"] = path
                except Exception:
                    pass

    log.info("Training done. Best ADE: %.4f @ %s", best_ade, best_ckpt)
    if run:
        run.finish()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--dino-repo-dir", default=None)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)
    log.info("Config: %s", cfg)
    train(cfg, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
