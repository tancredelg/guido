"""
predict_v5.py — inference and visualisation for DrivingPlannerV4.

Usage:
    uv run src/predict_v5.py --checkpoint checkpoints/best.pth --split val
    uv run src/predict_v5.py --checkpoint checkpoints/best.pth --split test --output submission.csv
    uv run src/predict_v5.py --checkpoint checkpoints/best.pth --split val --visualize
    uv run src/predict_v5.py --checkpoint checkpoints/best.pth --split val --visualize --tta
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from guido.dataset   import DrivingDataset, make_test_dataset, _sorted_pkl_files
from guido.model_v4  import DrivingPlannerV4
from guido.aux_heads import AuxDecoder, DepthHead, SegHead, visualize_aux
from guido.utils     import build_submission_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("cfg", {})

    model = DrivingPlannerV4(
        dino_model     = cfg.get("dino_model",      "dinov3_vitb16"),
        dino_repo_dir  = cfg["dino_repo_dir"],
        dino_weights   = cfg["dino_weights"],
        unfreeze_blocks= cfg.get("unfreeze_blocks", 1),
        d              = cfg.get("d",               384),
        num_heads      = cfg.get("num_heads",       4),
        hist_layers    = cfg.get("hist_layers",     2),
        dec_layers     = cfg.get("dec_layers",      3),
        K              = cfg.get("K",               1),
        n_coarse       = cfg.get("n_coarse",        6),
        dropout        = 0.0,
        use_cmd        = cfg.get("use_cmd",         False),
        rope_base      = cfg.get("rope_base",       10000.0),
        scene_fusion   = cfg.get("scene_fusion",    "rope_xattn"),
        causal_decoder = cfg.get("causal_decoder",  False),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if any("aux" not in k and "router" not in k for k in missing):
        log.warning("Unexpected missing non-aux keys: %s", missing)
    model.to(device).eval()
    log.info("Loaded trajectory model — epoch %d  val ADE %.4f",
             ckpt.get("epoch", -1), ckpt.get("val_ade", float("nan")))

    # Load aux heads if present in checkpoint
    aux_decoder = depth_head = seg_head = None
    dino_dim = model.backbone.embed_dim
    if "aux_decoder" in ckpt:
        d_dec      = cfg.get("aux_d_dec", 192)
        img_h      = cfg.get("img_h",     200)
        img_w      = cfg.get("img_w",     300)
        num_classes= cfg.get("num_seg_classes", 14)
        aux_decoder = AuxDecoder(dino_dim, d_dec, img_h, img_w).to(device)
        aux_decoder.load_state_dict(ckpt["aux_decoder"])
        aux_decoder.eval()
        if "depth_head" in ckpt:
            depth_head = DepthHead(d_dec).to(device)
            depth_head.load_state_dict(ckpt["depth_head"])
            depth_head.eval()
        if "seg_head" in ckpt:
            seg_head = SegHead(d_dec, num_classes).to(device)
            seg_head.load_state_dict(ckpt["seg_head"])
            seg_head.eval()
        log.info("Loaded aux decoder (depth=%s  seg=%s)",
                 depth_head is not None, seg_head is not None)

    return model, aux_decoder, depth_head, seg_head, cfg


# ── TTA ───────────────────────────────────────────────────────────────────────

def _mirror_batch(camera, history):
    cam_flip  = torch.flip(camera, dims=[-1])
    hist_flip = history.clone()
    hist_flip[:, :, 0] *= -1   # flip x coordinate
    hist_flip[:, :, 2] *= -1   # flip sin(heading)
    return cam_flip, hist_flip


@torch.no_grad()
def predict_batch(model, camera, history, command, tta=False):
    fine_preds, _, router_logits = model(camera, history, command)
    best_k = router_logits.argmax(dim=-1)
    stacked = torch.stack(fine_preds, dim=0)
    idx = best_k.view(1, -1, 1, 1).expand(1, -1, fine_preds[0].size(1), 2)
    pred = stacked.gather(0, idx).squeeze(0)

    if tta:
        cam_f, hist_f = _mirror_batch(camera, history)
        fp_f, _, rl_f = model(cam_f, hist_f, command)
        bk_f = rl_f.argmax(dim=-1)
        st_f = torch.stack(fp_f, dim=0)
        id_f = bk_f.view(1, -1, 1, 1).expand_as(idx)
        pred_f = st_f.gather(0, id_f).squeeze(0)
        pred_f[:, :, 0] *= -1   # un-flip x
        pred = (pred + pred_f) * 0.5

    return pred


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="val",
                        choices=["val", "test", "train"])
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--output",     default=None,
                        help="CSV output path (default: submission_v5.csv for test)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta",        action="store_true")
    parser.add_argument("--visualize",  action="store_true",
                        help="Save trajectory + aux predictions PDF")
    parser.add_argument("--vis-output", default="predictions_v5.pdf")
    parser.add_argument("--vis-n",      type=int, default=8)
    parser.add_argument("--num-workers",type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  TTA: %s", device, args.tta)

    model, aux_decoder, depth_head, seg_head, cfg = load_model(args.checkpoint, device)
    load_aux = aux_decoder is not None and args.visualize

    # Dataset
    data_dir = args.data_dir
    is_test  = args.split == "test"
    if is_test:
        files = _sorted_pkl_files(f"{data_dir}/test_public_real")
    else:
        files = _sorted_pkl_files(f"{data_dir}/val_real") if args.split == "val" else _sorted_pkl_files(f"{data_dir}/{args.split}")

    ds = DrivingDataset(files, test=is_test, augment=False, load_aux=load_aux)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_preds = []
    vis_cam, vis_dgt, vis_sgt, vis_dp, vis_sp, vis_fgt, vis_fp, vis_hist = (
        [], [], [], [], [], [], [], []
    )
    n_vis_collected = 0

    with torch.no_grad():
        for batch in loader:
            camera  = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)

            pred = predict_batch(model, camera, history, command, tta=args.tta)
            all_preds.append(pred.cpu().numpy())

            # Collect visualisation samples from first few batches
            if args.visualize and n_vis_collected < args.vis_n:
                n = min(args.vis_n - n_vis_collected, camera.size(0))
                vis_cam.append(camera[:n].cpu())
                vis_hist.append(history[:n].cpu())
                if "future" in batch:
                    vis_fgt.append(batch["future"][:n])
                vis_fp.append(pred[:n].cpu())

                if aux_decoder is not None:
                    af = aux_decoder(model.aux_stage_tokens())
                    if depth_head is not None:
                        vis_dp.append(depth_head(af)[:n].cpu())
                    if seg_head is not None:
                        vis_sp.append(seg_head(af)[:n].cpu())
                    if load_aux:
                        if "depth" in batch:
                            vis_dgt.append(batch["depth"][:n])
                        if "semantic_label" in batch:
                            vis_sgt.append(batch["semantic_label"][:n])

                n_vis_collected += n

    predictions = np.concatenate(all_preds, axis=0)   # (N, 60, 2)

    if is_test or args.output:
        out = args.output or "submission_v5.csv"
        build_submission_csv(predictions, out)

    if not is_test:
        # Compute ADE/FDE against GT
        from guido.dataset import DrivingDataset as _D
        gt_ds = DrivingDataset(files, test=False, augment=False)
        gt_loader = DataLoader(gt_ds, batch_size=256, shuffle=False, num_workers=2)
        gts = np.concatenate([b["future"].numpy() for b in gt_loader], axis=0)
        diff = predictions[:, :, :2] - gts[:, :, :2]
        per_step = np.linalg.norm(diff, axis=-1)   # (N, 60)
        ade = per_step.mean()
        fde = per_step[:, -1].mean()
        log.info("ADE: %.4f  FDE: %.4f", ade, fde)

    if args.visualize and vis_cam:
        cat = lambda lst: torch.cat(lst, dim=0) if lst else None
        visualize_aux(
            camera    = cat(vis_cam),
            depth_gt  = cat(vis_dgt) if vis_dgt else None,
            seg_gt    = cat(vis_sgt) if vis_sgt else None,
            depth_pred= cat(vis_dp)  if vis_dp  else None,
            seg_pred  = cat(vis_sp)  if vis_sp  else None,
            traj_gt   = cat(vis_fgt) if vis_fgt else None,
            traj_pred = cat(vis_fp),
            history   = cat(vis_hist),
            output_path = args.vis_output,
            n = args.vis_n,
        )
        log.info("Saved → %s", args.vis_output)


if __name__ == "__main__":
    main()