"""
Miscellaneous utilities: submission CSV, checkpoint names, reproducibility.
"""

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ── Reproducibility ────────────────────────────────────────────────────────────


def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Trades a small amount of speed for determinism on CUDA ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Checkpoints ────────────────────────────────────────────────────────────────


def checkpoint_path(checkpoint_dir: str, epoch: int, val_ade: float) -> str:
    """
    Return a descriptive checkpoint filename that encodes the timestamp,
    epoch, and validation ADE so that a directory listing is self-documenting.

    Example: checkpoints/run_20260417_1432_epoch042_ade1.8731.pth
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"run_{ts}_epoch{epoch:03d}_ade{val_ade:.4f}.pth"
    return os.path.join(checkpoint_dir, name)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_ade: float,
    cfg: dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_ade": val_ade,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Load weights into model in-place; return the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


# ── Submission CSV ─────────────────────────────────────────────────────────────


def build_submission_csv(predictions: np.ndarray, output_path: str) -> None:
    """
    Build a Kaggle submission CSV from a predictions array.

    Args:
        predictions : float array of shape (N, T, 2) – predicted (x, y).
        output_path : where to write the CSV.

    Output format (matches the leaderboard spec):
        id, x_1, y_1, x_2, y_2, ..., x_60, y_60
    """
    N, T, _ = predictions.shape  # e.g. (1000, 60, 2)
    flat = predictions.reshape(N, T * 2)  # row-major → [x1,y1,x2,y2,...]

    col_names = ["id"]
    for t in range(1, T + 1):
        col_names += [f"x_{t}", f"y_{t}"]

    df = pd.DataFrame(flat, columns=col_names[1:])
    df.insert(0, "id", np.arange(N))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {N} predictions → {output_path}")
