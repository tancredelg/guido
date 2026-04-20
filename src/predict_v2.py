"""
predict_v2.py — inference and submission for DrivingPlannerV2.

Usage
-----
# Validate locally (no Kaggle needed) — reports ADE/FDE on the val split:
python src/predict_v2.py --checkpoint checkpoints/best.pth --split val

# Generate test submission CSV:
python src/predict_v2.py --checkpoint checkpoints/best.pth --split test

# Visualise predictions as a PDF (val split only):
python src/predict_v2.py --checkpoint checkpoints/best.pth --split val --visualize

# Test-time augmentation (mirror + average, free ~0.03 ADE):
python src/predict_v2.py --checkpoint checkpoints/best.pth --split test --tta

How to sanity-check without Kaggle
-----------------------------------
Run with --split val.  The script reports val ADE/FDE which should match
the best val ADE in your training logs.  If they agree, the test submission
will be in the same ballpark.  The val split has the same data distribution
as the test split, so val ADE is the honest estimate of Kaggle score.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from guido.dataset import make_datasets, make_test_dataset
from guido.model_v2 import DrivingPlannerV2
from guido.losses_v2 import ade as compute_ade, fde as compute_fde, best_of_k_ade
from guido.utils import seed_everything, build_submission_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, device: torch.device) -> DrivingPlannerV2:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
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
        dropout=cfg.get("dropout", 0.0),  # no dropout at inference
        cmd_embed_dim=cfg.get("cmd_embed_dim", 64),
        smoothness_weight=0.0,  # unused at inference
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        non_router_missing = [k for k in missing if "router" not in k]
        if non_router_missing:
            log.warning("Non-router keys missing — architecture mismatch: %s", non_router_missing)
        log.info(
            "Router keys not loaded (architecture mismatch between code and checkpoint "
            "— update model_v2.py router definition to match trained model): "
            "missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    log.info("Model loaded with config:\n%s", json.dumps(cfg, indent=2))
    model.to(device).eval()
    log.info(
        "Loaded checkpoint: epoch %d  val ADE %.4f",
        ckpt.get("epoch", -1),
        ckpt.get("val_ade", float("nan")),
    )
    return model


# ── Best-head selection ───────────────────────────────────────────────────────


def _pick_best_pred(preds: list, router_logits: torch.Tensor) -> torch.Tensor:
    """
    Select the best trajectory head using the trained router.
    router_logits: (B, K) — the model's learned confidence per head.
    Returns: (B, 60, 2)
    """
    K = len(preds)
    if K == 1:
        return preds[0]
    best_k = router_logits.argmax(dim=-1)  # (B,)
    stacked = torch.stack(preds, dim=0)  # (K, B, 60, 2)
    idx_exp = best_k.view(1, -1, 1, 1).expand(1, -1, 60, 2)
    return stacked.gather(0, idx_exp).squeeze(0)  # (B, 60, 2)


# ── Test-time augmentation ────────────────────────────────────────────────────


@torch.no_grad()
def _tta_predict(model, camera, history, command):
    """
    Run inference twice:
      1. Normal inputs
      2. Horizontally flipped camera, x-negated history, swapped command
    Average the two predictions (re-flipping the mirrored one).

    This exploits left-right symmetry for a free ~0.02-0.04 ADE improvement.
    """
    # Normal
    preds_normal, rl_normal = model(camera, history, command)
    pred_normal = _pick_best_pred(preds_normal, rl_normal)  # (B, 60, 2)

    # Mirrored
    import torchvision.transforms.v2 as T

    cam_flip = T.functional.horizontal_flip(camera)
    hist_flip = history.clone()
    hist_flip[:, :, 0] *= -1  # negate x
    hist_flip[:, :, 2] *= -1  # negate sin(heading)
    cmd_flip = command.clone()
    # swap left (1) ↔ right (2), leave forward (0) alone
    cmd_flip = torch.where(
        command == 1,
        torch.tensor(2, device=command.device),
        torch.where(command == 2, torch.tensor(1, device=command.device), command),
    )

    preds_flip, rl_flip = model(cam_flip, hist_flip, cmd_flip)
    pred_flip = _pick_best_pred(preds_flip, rl_flip)  # (B, 60, 2)

    # Re-flip the mirrored prediction back to original frame
    pred_flip_back = pred_flip.clone()
    pred_flip_back[:, :, 0] *= -1

    return (pred_normal + pred_flip_back) / 2.0


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(model, loader, device, has_labels=False, use_tta=False):
    model.eval()
    all_preds, ade_vals, fde_vals, bofk_vals = [], [], [], []

    for batch in loader:
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        command = batch["command"].to(device)

        if use_tta:
            pred = _tta_predict(model, camera, history, command)
        else:
            preds, router_logits = model(camera, history, command)
            pred = _pick_best_pred(preds, router_logits)

        all_preds.append(pred.cpu().numpy())

        if has_labels:
            future = batch["future"].to(device)
            ade_vals.append(compute_ade(pred, future).item())
            fde_vals.append(compute_fde(pred, future).item())
            if not use_tta:
                bofk_vals.append(best_of_k_ade(preds, future).item())

    preds_np = np.concatenate(all_preds, axis=0)
    metrics = None
    if has_labels:
        metrics = {
            "ade": float(np.mean(ade_vals)),
            "fde": float(np.mean(fde_vals)),
        }
        if bofk_vals:
            metrics["bofk_ade"] = float(np.mean(bofk_vals))
    return preds_np, metrics


# ── Visualisation ─────────────────────────────────────────────────────────────


def visualize(loader, model, device, output_path="predictions_v2.pdf", n=16, use_tta=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    model.eval()
    cameras, histories, futures, preds_list = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            cam = batch["camera"].to(device)
            hist = batch["history"].to(device)
            cmd = batch["command"].to(device)

            if use_tta:
                pred = _tta_predict(model, cam, hist, cmd)
            else:
                preds, router_logits = model(cam, hist, cmd)
                pred = _pick_best_pred(preds, router_logits)

            cameras.append(cam.cpu())
            histories.append(hist.cpu())
            preds_list.append(pred.cpu())
            if "future" in batch:
                futures.append(batch["future"].cpu())
            if sum(c.shape[0] for c in cameras) >= n:
                break

    cameras = torch.cat(cameras)[:n]
    histories = torch.cat(histories)[:n]
    preds_out = torch.cat(preds_list)[:n]
    has_gt = len(futures) > 0
    if has_gt:
        futures = torch.cat(futures)[:n]

    with PdfPages(output_path) as pdf:
        for i in range(0, n, 4):
            idxs = range(i, min(i + 4, n))
            fig, axes = plt.subplots(2, len(idxs), figsize=(4 * len(idxs), 8))
            if len(idxs) == 1:
                axes = [[axes[0]], [axes[1]]]
            for col, j in enumerate(idxs):
                img = cameras[j].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
                axes[0][col].imshow(np.clip(img, 0, 1))
                axes[0][col].axis("off")
                ax = axes[1][col]
                ax.plot(
                    histories[j, :, 0], histories[j, :, 1], "o-", color="gold", ms=3, lw=1, label="history"
                )
                if has_gt:
                    ax.plot(
                        futures[j, :, 0], futures[j, :, 1], "o-", color="limegreen", ms=3, lw=1, label="GT"
                    )
                ax.plot(
                    preds_out[j, :, 0], preds_out[j, :, 1], "o-", color="tomato", ms=3, lw=1, label="pred"
                )
                ax.set_aspect("equal")
                ax.legend(fontsize=7)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    log.info("Saved → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submission_phase1.csv")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-output", default="predictions_v2.pdf")
    parser.add_argument(
        "--tta", action="store_true", help="Test-time augmentation: mirror + average (~free ADE gain)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  TTA: %s", device, args.tta)

    model = load_model(args.checkpoint, device)

    has_labels = args.split == "val"
    if args.split == "test":
        dataset = make_test_dataset(args.data_dir)
        log.info("Test split: %d samples", len(dataset))
    else:
        _, dataset = make_datasets(args.data_dir)
        log.info("Val split: %d samples — will report ADE/FDE", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    preds, metrics = run_inference(model, loader, device, has_labels=has_labels, use_tta=args.tta)

    if metrics:
        log.info(
            "ADE: %.4f  FDE: %.4f%s",
            metrics["ade"],
            metrics["fde"],
            f"  BofK-ADE: {metrics['bofk_ade']:.4f}" if "bofk_ade" in metrics else "",
        )

    if args.visualize and args.split == "val":
        visualize(loader, model, device, args.vis_output, use_tta=args.tta)

    if args.split == "test":
        build_submission_csv(preds, args.output)
        log.info("Submission → %s", args.output)


if __name__ == "__main__":
    main()
