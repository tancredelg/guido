import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def checkpoint_path(checkpoint_dir: str, epoch: int, val_ade: float) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"run_{ts}_epoch{epoch:03d}_ade{val_ade:.4f}.pth"
    return os.path.join(checkpoint_dir, name)


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_ade, cfg):
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
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def build_submission_csv(predictions: np.ndarray, output_path: str) -> None:
    N, T, _ = predictions.shape
    flat = predictions.reshape(N, T * 2)
    cols = ["id"] + [f"{c}_{t}" for t in range(1, T + 1) for c in ("x", "y")]
    df = pd.DataFrame(flat, columns=cols[1:])
    df.insert(0, "id", np.arange(N))
    df.columns = cols
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {N} predictions → {output_path}")
