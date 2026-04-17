"""
train.py – main training entry point for Guido (Phase 1).

Usage:
    python src/train.py                          # uses configs/baseline.yaml
    python src/train.py --config configs/exp.yaml
    python src/train.py --config configs/baseline.yaml --data-dir /scratch/data
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

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from guido import (
    DrivingPlanner,
    make_datasets,
    huber_loss,
    ade as compute_ade,
    fde as compute_fde,
    seed_everything,
    save_checkpoint,
    checkpoint_path,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI flags override yaml values."""
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.epochs:
        cfg["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.dino_repo_dir:
        cfg["dino_repo_dir"] = args.dino_repo_dir
    if args.dino_weights:
        cfg["dino_weights"] = args.dino_weights
    return cfg


# ── Validation ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model: DrivingPlanner, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    ade_vals, fde_vals = [], []

    for batch in loader:
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        command = batch["command"].to(device)
        future = batch["future"].to(device)

        pred = model(camera, history, command)  # (B, 60, 2)

        total_loss += huber_loss(pred, future).item()
        ade_vals.append(compute_ade(pred, future).item())
        fde_vals.append(compute_fde(pred, future).item())

    return {
        "val_loss": total_loss / len(loader),
        "val_ade": float(np.mean(ade_vals)),
        "val_fde": float(np.mean(fde_vals)),
    }


# ── Training loop ──────────────────────────────────────────────────────────────


def train(cfg: dict) -> None:
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds, val_ds = make_datasets(cfg["data_dir"])
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
        dino_model=cfg.get("dino_model", "dinov3_vits16"),
        dino_repo_dir=cfg["dino_repo_dir"],
        dino_weights=cfg["dino_weights"],
        hist_hidden_dim=cfg.get("hist_hidden_dim", 128),
        cmd_embed_dim=cfg.get("cmd_embed_dim", 32),
        fusion_dim=cfg.get("fusion_dim", 256),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    log.info(
        "Model ready – trainable params: %s",
        f"{model.num_trainable_params():,}",
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use AdamW with weight decay only on non-bias / non-norm parameters.
    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])
    ]
    no_decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.get("weight_decay", 1e-4)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.get("lr", 3e-4),
    )

    # ── Scheduler: cosine annealing from lr → min_lr ─────────────────────────
    num_epochs = cfg["num_epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=cfg.get("min_lr", 1e-6),
    )

    # ── Optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_ade = float("inf")
    best_ckpt = None

    if cfg.get("resume"):
        log.info("Resuming from %s", cfg["resume"])
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ade = ckpt["val_ade"]
        log.info("Resumed at epoch %d (best ADE so far: %.4f)", start_epoch, best_ade)

    grad_clip = cfg.get("grad_clip", 1.0)
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    log_interval = cfg.get("log_interval", 20)  # batches

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
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
            loss = huber_loss(pred, future)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)

            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % log_interval == 0:
                log.info(
                    "  epoch %d  step %d/%d  loss %.4f",
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | lr %.2e | train_loss %.4f | val_loss %.4f | ADE %.4f | FDE %.4f | %.0fs",
            epoch + 1,
            num_epochs,
            scheduler.get_last_lr()[0],
            train_loss / len(train_loader),
            metrics["val_loss"],
            metrics["val_ade"],
            metrics["val_fde"],
            elapsed,
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        if metrics["val_ade"] < best_ade:
            best_ade = metrics["val_ade"]
            path = checkpoint_path(ckpt_dir, epoch + 1, best_ade)
            save_checkpoint(path, model, optimizer, scheduler, epoch + 1, best_ade, cfg)
            log.info("  ✓ New best ADE %.4f → saved to %s", best_ade, path)
            best_ckpt = path

    log.info("Training done. Best ADE: %.4f @ %s", best_ade, best_ckpt)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Guido driving planner")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--data-dir", default=None, help="Override data_dir in config")
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--dino-repo-dir", default=None, help="Path to local dinov3 repo clone")
    parser.add_argument("--dino-weights", default=None, help="Path to downloaded .pth weights file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)

    log.info("Config: %s", cfg)
    train(cfg)


if __name__ == "__main__":
    main()
