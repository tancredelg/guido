"""
predict.py – inference and submission generation.

Usage:
    # Test submission:
    python src/predict.py --checkpoint checkpoints/best.pth

    # Val sanity-check with trajectory PDF:
    python src/predict.py --checkpoint checkpoints/best.pth --split val --visualize
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from guido import (
    DrivingPlanner,
    make_test_dataset,
    make_datasets,
    load_checkpoint,
    build_submission_csv,
    seed_everything,
    ade as compute_ade,
    fde as compute_fde,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def visualize_predictions(loader, model, device, output_path="predictions.pdf", n=16):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    model.eval()
    cameras, histories, futures, preds = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            cam = batch["camera"].to(device)
            hist = batch["history"].to(device)
            cmd = batch["command"].to(device)
            pred = model(cam, hist, cmd).cpu()

            cameras.append(cam.cpu())
            histories.append(hist.cpu())
            preds.append(pred)
            if "future" in batch:
                futures.append(batch["future"].cpu())
            if sum(c.shape[0] for c in cameras) >= n:
                break

    cameras = torch.cat(cameras)[:n]
    histories = torch.cat(histories)[:n]
    preds = torch.cat(preds)[:n]
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
                ax.plot(preds[j, :, 0], preds[j, :, 1], "o-", color="tomato", ms=3, lw=1, label="pred")
                ax.set_aspect("equal")
                ax.legend(fontsize=7)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    log.info("Saved → %s", output_path)


@torch.no_grad()
def run_inference(model, loader, device, has_labels=False):
    model.eval()
    all_preds, ade_vals, fde_vals = [], [], []

    for batch in loader:
        camera = batch["camera"].to(device)
        history = batch["history"].to(device)
        command = batch["command"].to(device)
        pred = model(camera, history, command)
        all_preds.append(pred.cpu().numpy())

        if has_labels:
            future = batch["future"].to(device)
            ade_vals.append(compute_ade(pred, future).item())
            fde_vals.append(compute_fde(pred, future).item())

    preds = np.concatenate(all_preds, axis=0)
    metrics = {"ade": float(np.mean(ade_vals)), "fde": float(np.mean(fde_vals))} if has_labels else None
    return preds, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submission_phase1.csv")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-output", default="predictions.pdf")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg", {})

    model = DrivingPlanner(
        dino_model=saved_cfg.get("dino_model", "dinov3_vits16"),
        dino_repo_dir=saved_cfg["dino_repo_dir"],
        dino_weights=saved_cfg["dino_weights"],
        hist_hidden_dim=saved_cfg.get("hist_hidden_dim", 128),
        cmd_embed_dim=saved_cfg.get("cmd_embed_dim", 32),
        fusion_dim=saved_cfg.get("fusion_dim", 256),
        dropout=saved_cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    log.info("Loaded epoch %d  val ADE %.4f", ckpt.get("epoch", -1), ckpt.get("val_ade", float("nan")))

    has_labels = args.split == "val"
    dataset = make_test_dataset(args.data_dir) if args.split == "test" else make_datasets(args.data_dir)[1]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    preds, metrics = run_inference(model, loader, device, has_labels=has_labels)
    if metrics:
        log.info("ADE: %.4f  FDE: %.4f", metrics["ade"], metrics["fde"])

    if args.visualize and args.split == "val":
        visualize_predictions(loader, model, device, output_path=args.vis_output)

    if args.split == "test":
        build_submission_csv(preds, args.output)


if __name__ == "__main__":
    main()
