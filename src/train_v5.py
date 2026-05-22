"""
train_v5.py — Phase 3 sim-to-real fine-tuning on mixed synthetic + real data.

Usage:
    uv run src/train_v5.py --config configs/V5/phase3.yaml
    uv run src/train_v5.py --config configs/V5/phase3.yaml --no-wandb
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

from guido.dataset    import make_datasets, make_datasets_mixed, make_test_dataset
from guido.model_v4   import DrivingPlannerV4

from guido.losses_v2  import winner_takes_all_loss, ade, fde, best_of_k_ade
from guido.utils      import (
    seed_everything, save_checkpoint, load_checkpoint,
    checkpoint_path, build_submission_csv,
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
    for k, v in [("data_dir", args.data_dir),
                 ("dino_repo_dir", args.dino_repo_dir),
                 ("dino_weights",  args.dino_weights)]:
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
            project = cfg.get("wandb_project", "dlav-guido"),
            entity  = cfg.get("wandb_entity",  "tancredelg-personal"),
            name    = cfg.get("wandb_run_name", None),
            config  = cfg,
            resume  = "allow",
        )
        log.info("W&B: %s", run.url)
        return run
    except Exception as e:
        log.warning("W&B init failed (%s)", e)
        return None

def wandb_log(run, step, d):
    if run:
        try: run.log(d, step=step)
        except Exception: pass


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
        fut  = data["sdc_future_feature"]
        disp = float(np.linalg.norm(fut[-1, :2] - fut[0, :2]))
        displacements.append(max(disp, 0.5))   # floor so stationary isn't zero
    weights = torch.tensor(displacements, dtype=torch.float)
    weights = weights / weights.sum()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    ade_vals, fde_vals, bofk_vals = [], [], []
    cmd_ade   = {0: [], 1: [], 2: []}
    horizon   = {"early": [], "mid": [], "far": []}
    static_ade, moving_ade = [], []
    vis_batch = None

    sw = cfg.get("smoothness_weight", 0.1)

    for batch in loader:
        camera  = batch["camera"].to(device)
        history = batch["history"].to(device)
        command = batch["command"].to(device)
        future  = batch["future"].to(device)

        fine_preds, coarse_preds, router_logits = model(camera, history, command)

        loss, _ = winner_takes_all_loss(
            fine_preds, coarse_preds, router_logits, future,
            smoothness_weight = sw,
            coarse_weight     = cfg.get("coarse_weight",     0.3),
            wta_weight        = cfg.get("wta_weight",        0.8),
            router_weight     = cfg.get("router_weight",     0.5),
            fde_weight        = cfg.get("fde_weight",        0.0),
            stationary_weight = cfg.get("stationary_weight", 0.0),
            weighting         = cfg.get("loss_weighting",    "linear"),
            near_weight       = cfg.get("loss_near_weight",  0.5),
            far_weight        = cfg.get("loss_far_weight",   2.0),
            router_active     = True,
        )
        total_loss += loss.item()

        # Head selection via trained router
        stacked   = torch.stack(fine_preds, dim=0)
        gt_xy     = future[..., :2]
        best_k    = router_logits.argmax(dim=-1)
        idx_exp   = best_k.view(1, -1, 1, 1).expand(1, -1, 60, 2)
        best_pred = stacked.gather(0, idx_exp).squeeze(0)

        per_step = torch.norm(best_pred - gt_xy, p=2, dim=-1)  # (B, 60)

        ade_vals.append(per_step.mean().item())
        fde_vals.append(per_step[:, -1].mean().item())
        bofk_vals.append(best_of_k_ade(fine_preds, future).item())

        for c in range(3):
            m = command == c
            if m.any():
                cmd_ade[c].append(per_step[m].mean().item())

        horizon["early"].append(per_step[:, :20].mean().item())
        horizon["mid"].append(per_step[:, 20:40].mean().item())
        horizon["far"].append(per_step[:, 40:].mean().item())

        disp = torch.norm(future[:, -1, :2] - future[:, 0, :2], p=2, dim=-1)
        sm   = disp < 2.0
        if sm.any():  static_ade.append(per_step[sm].mean().item())
        if (~sm).any(): moving_ade.append(per_step[~sm].mean().item())

        if vis_batch is None:
            vis_batch = {k: v.cpu() for k, v in batch.items()}
            vis_batch["_pred"] = best_pred.cpu()

    CMD = {0: "forward", 1: "left", 2: "right"}
    metrics = {
        "val/loss":      total_loss / len(loader),
        "val/ade":       float(np.mean(ade_vals)),
        "val/fde":       float(np.mean(fde_vals)),
        "val/bofk_ade":  float(np.mean(bofk_vals)),
        "val/ade_early": float(np.mean(horizon["early"])),
        "val/ade_mid":   float(np.mean(horizon["mid"])),
        "val/ade_far":   float(np.mean(horizon["far"])),
    }
    for c, name in CMD.items():
        if cmd_ade[c]:
            metrics[f"val/ade_{name}"] = float(np.mean(cmd_ade[c]))
    if static_ade: metrics["val/ade_static"] = float(np.mean(static_ade))
    if moving_ade: metrics["val/ade_moving"] = float(np.mean(moving_ade))

    return metrics, vis_batch


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, use_wandb: bool) -> None:
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Datasets ──────────────────────────────────────────────────────────
    use_real_data = cfg.get("use_real_data", True)
    if use_real_data:
        train_ds, val_ds = make_datasets_mixed(
            cfg["data_dir"],
            real_train_frac = cfg.get("real_train_frac", 0.8),
            mirror_p        = cfg.get("mirror_p",        0.2),
            hist_noise_std  = cfg.get("hist_noise_std",  0.01),
            hist_dropout_p  = cfg.get("hist_dropout_p",  0.1),
            mirror_warmup   = cfg.get("mirror_warmup",   5),
        )
    else:
        train_ds, val_ds = make_datasets(
            cfg["data_dir"],
            mirror_p       = cfg.get("mirror_p",        0.2),
            hist_noise_std = cfg.get("hist_noise_std",  0.01),
            hist_dropout_p = cfg.get("hist_dropout_p",  0.1),
            mirror_warmup  = cfg.get("mirror_warmup",   5),
        )
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # Optionally use trajectory-length weighted sampling
    sampler    = None
    shuffle    = True
    use_sampler = cfg.get("weighted_sampler", False)
    if use_sampler:
        log.info("Using trajectory-length weighted sampler")
        sampler = make_weighted_sampler(train_ds)
        shuffle = False   # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = DrivingPlannerV4(
        dino_model        = cfg.get("dino_model",         "dinov3_vitb16"),
        dino_repo_dir     = cfg["dino_repo_dir"],
        dino_weights      = cfg["dino_weights"],
        unfreeze_blocks   = cfg.get("unfreeze_blocks",    2),
        d                 = cfg.get("d",                  384),
        num_heads         = cfg.get("num_heads",          4),
        hist_layers       = cfg.get("hist_layers",        2),
        dec_layers        = cfg.get("dec_layers",         3),
        K                 = cfg.get("K",                  1),
        n_coarse          = cfg.get("n_coarse",           6),
        dropout           = cfg.get("dropout",            0.05),
        smoothness_weight = cfg.get("smoothness_weight",  0.05),
        cmd_embed_dim     = cfg.get("cmd_embed_dim",      64),
        rope_base         = cfg.get("rope_base",          10000.0),
        use_cmd           = cfg.get("use_cmd",            False),
        scene_fusion      = cfg.get("scene_fusion",       "rope_xattn"),
        causal_decoder    = cfg.get("causal_decoder",     False),
    ).to(device)
    log.info("Trajectory model trainable params: %s", f"{model.num_trainable_params():,}")

    # ── Auxiliary perception decoder ──────────────────────────────────────
    # Built only when at least one aux task is enabled in config.
    use_depth = cfg.get("use_depth_aux", True)
    use_seg   = cfg.get("use_seg_aux",   True)
    log.info("Aux tasks — depth: %s  seg: %s", use_depth, use_seg)
    num_seg_classes = cfg.get("num_seg_classes", 20)   # set after data exploration
    dino_dim  = model.backbone.embed_dim
    img_h     = cfg.get("img_h", 200)
    img_w     = cfg.get("img_w", 300)
    d_dec     = cfg.get("aux_d_dec", 128)

    # DINOv3 ViT-B/16 patch grid depends on input size.
    # Images are resized to DINO_INPUT_SIZE (256) before the backbone,
    # so the patch grid is always 256//16 = 16×16 = 256 tokens.
    # The aux decoder output is then bilinearly upsampled to img_h×img_w.
    dino_input = 256   # DINO_INPUT_SIZE from dataset.py
    patch_size = 16    # ViT-B/16
    grid_h = grid_w = dino_input // patch_size   # always 16×16

    aux_decoder, depth_head, seg_head = None, None, None
    use_normals = cfg.get("use_normals_aux", True)
    if use_depth or use_seg:
        aux_decoder = AuxDecoder(
            d_backbone = dino_dim,
            d_dec      = cfg.get("aux_d_dec", 192),
            out_h      = img_h,
            out_w      = img_w,
        ).to(device)
        if use_depth:
            depth_head = DepthHead(cfg.get("aux_d_dec", 192)).to(device)
        if use_seg:
            seg_head = SegHead(cfg.get("aux_d_dec", 192), num_seg_classes).to(device)
        aux_params = (list(aux_decoder.parameters())
                      + (list(depth_head.parameters()) if depth_head else [])
                      + (list(seg_head.parameters())   if seg_head   else []))
        log.info("Aux decoder BUILT — params: %s", f"{sum(p.numel() for p in aux_params):,}")
    else:
        aux_params  = []
        use_normals = False
        log.warning("Aux decoder NOT built — use_depth_aux and use_seg_aux both False")

    # ── Optimiser ─────────────────────────────────────────────────────────
    base_lr     = cfg.get("lr",          1e-4)
    backbone_lr = cfg.get("backbone_lr", 8e-6)
    aux_lr      = cfg.get("aux_lr",      2e-4)
    wd          = cfg.get("weight_decay", 1e-3)

    def no_wd(n): return "bias" in n or "norm" in n or "pe" in n

    head_decay    = [p for n, p in model.named_parameters()
                     if p.requires_grad and not n.startswith("backbone.") and not no_wd(n)]
    head_no_decay = [p for n, p in model.named_parameters()
                     if p.requires_grad and not n.startswith("backbone.") and no_wd(n)]
    bb_decay      = [p for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith("backbone.") and not no_wd(n)]
    bb_no_decay   = [p for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith("backbone.") and no_wd(n)]

    groups = [
        {"params": head_decay,    "lr": base_lr,     "weight_decay": wd},
        {"params": head_no_decay, "lr": base_lr,     "weight_decay": 0.0},
    ]
    if aux_params:
        groups.append({"params": aux_params, "lr": aux_lr, "weight_decay": wd})
    if bb_decay or bb_no_decay:
        groups += [
            {"params": bb_decay,    "lr": backbone_lr, "weight_decay": wd},
            {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        ]
    log.info("Head lr %.2e  Backbone lr %.2e  Aux lr %.2e", base_lr, backbone_lr, aux_lr)

    optimizer = optim.AdamW(groups)

    num_epochs    = cfg["num_epochs"]
    warmup_epochs = cfg.get("warmup_epochs", 20)
    # Cosine with warm restarts every T_0 epochs — reduces the plateau spikes
    # seen with plain cosine, because the LR periodically resets and reescapes
    # local optima rather than decaying monotonically into one.
    T_0 = cfg.get("lr_restart_epochs", max(num_epochs // 4, 50))
    cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1,
        eta_min=cfg.get("min_lr", 5e-7),
    )
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )

    # ── Resume or warm-start from Phase 1 checkpoint ─────────────────────
    start_epoch = 0
    best_ade    = float("inf")
    best_ckpt   = None
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ade    = ckpt["val_ade"]
        log.info("Resumed epoch %d (best ADE %.4f)", start_epoch, best_ade)
    elif cfg.get("warm_start"):
        ckpt = torch.load(cfg["warm_start"], map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        non_aux_missing = [k for k in missing if "aux" not in k and "router" not in k]
        if non_aux_missing:
            log.warning("Warm-start: unexpected missing keys: %s", non_aux_missing)
        log.info("Warm-start from %s — missing %d model keys", cfg["warm_start"], len(missing))
        # Restore aux decoder if present in checkpoint
        if aux_decoder is not None and "aux_decoder" in ckpt:
            aux_decoder.load_state_dict(ckpt["aux_decoder"])
            log.info("  Loaded aux_decoder weights from checkpoint")
        if depth_head is not None and "depth_head" in ckpt:
            depth_head.load_state_dict(ckpt["depth_head"])
            log.info("  Loaded depth_head weights from checkpoint")
        if seg_head is not None and "seg_head" in ckpt:
            seg_head.load_state_dict(ckpt["seg_head"])
            log.info("  Loaded seg_head weights from checkpoint")

    run         = init_wandb(cfg, enabled=use_wandb)
    grad_clip   = cfg.get("grad_clip",    1.0)
    ckpt_dir    = cfg.get("checkpoint_dir", "checkpoints")
    log_every   = cfg.get("log_interval",  20)
    global_step = start_epoch * len(train_loader)
    sw          = cfg.get("smoothness_weight", 0.1)
    K           = cfg.get("K", 1)
    router_warmup_epochs = cfg.get("router_warmup_epochs", 0)

    # Aux loss weights — ramped from 0 over aux_warmup_epochs to final value.
    # Ramp prevents aux losses swamping trajectory signal in early training
    # when the backbone features are not yet trajectory-specialised.
    lambda_depth_final = cfg.get("lambda_depth", 0.1)
    lambda_seg_final   = cfg.get("lambda_seg",   0.3)
    aux_warmup_epochs  = cfg.get("aux_warmup_epochs", 10)

    def aux_lambda(epoch_: int, final: float) -> float:
        if epoch_ >= aux_warmup_epochs:
            return final
        return final * epoch_ / max(aux_warmup_epochs, 1)

    # ── Main loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        router_frozen = (epoch < router_warmup_epochs) and K > 1
        if K > 1:
            for p in model.router.parameters():
                p.requires_grad = not router_frozen
        if epoch == router_warmup_epochs and K > 1:
            log.info("Epoch %d: router unfrozen", epoch + 1)
        wandb_log(run, epoch, {"train/router_active": int(not router_frozen)})

        lam_depth = aux_lambda(epoch, lambda_depth_final)
        lam_seg   = aux_lambda(epoch, lambda_seg_final)

        if aux_decoder is not None:
            aux_decoder.train()
        if depth_head is not None:
            depth_head.train()
        if seg_head is not None:
            seg_head.train()

        train_ds.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            camera  = batch["camera"].to(device)
            history = batch["history"].to(device)
            command = batch["command"].to(device)
            future  = batch["future"].to(device)
            depth_gt = batch.get("depth")
            seg_gt   = batch.get("semantic_label")
            if depth_gt is not None:
                depth_gt = depth_gt.to(device)
            if seg_gt is not None:
                seg_gt = seg_gt.to(device)

            optimizer.zero_grad(set_to_none=True)
            fine_preds, coarse_preds, router_logits = model(camera, history, command)
            loss, winner = winner_takes_all_loss(
                fine_preds, coarse_preds, router_logits, future,
                smoothness_weight = sw,
                coarse_weight     = cfg.get("coarse_weight",     0.3),
                wta_weight        = cfg.get("wta_weight",        0.8),
                router_weight     = cfg.get("router_weight",     0.5),
                fde_weight        = cfg.get("fde_weight",        0.0),
                stationary_weight = cfg.get("stationary_weight", 0.0),
                weighting         = cfg.get("loss_weighting",    "linear"),
                near_weight       = cfg.get("loss_near_weight",  0.5),
                far_weight        = cfg.get("loss_far_weight",   2.0),
                router_active     = not router_frozen,
            )

            # ── Auxiliary losses ──────────────────────────────────────────
            if aux_decoder is not None and (lam_depth > 0 or lam_seg > 0):
                stage_tokens = model.aux_stage_tokens()
                aux_feat     = aux_decoder(stage_tokens)

                l_depth = l_seg = l_normals = torch.tensor(0.0, device=device)

                if depth_head is not None and depth_gt is not None and lam_depth > 0:
                    pred_depth = depth_head(aux_feat)           # (B, 1, H, W)
                    if depth_gt.dim() == 4 and depth_gt.shape[-1] == 1:
                        depth_gt = depth_gt.permute(0, 3, 1, 2)
                    l_depth = depth_loss(pred_depth, depth_gt)
                    loss     = loss + lam_depth * l_depth
                    if use_normals:
                        l_normals = normals_loss(pred_depth, depth_gt)
                        loss      = loss + cfg.get("lambda_normals", 0.1) * l_normals

                if seg_head is not None and seg_gt is not None and lam_seg > 0:
                    pred_seg = seg_head(aux_feat)
                    if seg_gt.dim() == 4:
                        seg_gt = seg_gt.squeeze(-1)
                    l_seg = seg_loss(pred_seg, seg_gt,
                                     ignore_index=cfg.get("seg_ignore_index", 255))
                    loss = loss + lam_seg * l_seg
            else:
                l_depth = l_seg = l_normals = torch.tensor(0.0)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                if aux_decoder is not None:
                    torch.nn.utils.clip_grad_norm_(aux_decoder.parameters(), grad_clip)
            optimizer.step()

            train_loss  += loss.item()
            global_step += 1

            if (step + 1) % log_every == 0:
                w_counts = winner.bincount(minlength=cfg.get("K", 1)).float()
                w_frac   = w_counts / w_counts.sum()
                wandb_log(run, global_step, {
                    "train/loss_step":    loss.item(),
                    "train/loss_depth":   l_depth.item(),
                    "train/loss_seg":     l_seg.item(),
                    "train/lambda_depth": lam_depth,
                    "train/lambda_seg":   lam_seg,
                    **{f"train/head_{k}_win_frac": w_frac[k].item()
                       for k in range(len(w_frac))},
                })
                log.info("  epoch %d  step %d/%d  loss %.4f  "
                         "depth %.4f  seg %.4f",
                         epoch + 1, step + 1, len(train_loader),
                         loss.item(), l_depth.item(), l_seg.item())

                wandb_log(run, global_step, {
                    "train/loss_step":    loss.item(),
                    "train/loss_depth":   l_depth.item(),
                    "train/loss_seg":     l_seg.item(),
                    "train/loss_normals": l_normals.item(),
                    "train/lambda_depth": lam_depth,
                    "train/lambda_seg":   lam_seg,
                    **{f"train/head_{k}_win_frac": w_frac[k].item()
                       for k in range(len(w_frac))},
                })
                log.info("  epoch %d  step %d/%d  loss %.4f  depth %.4f  seg %.4f  norm %.4f",
                         epoch + 1, step + 1, len(train_loader),
                         loss.item(), l_depth.item(), l_seg.item(), l_normals.item())

        scheduler.step()

        metrics, vis_batch = validate(model, val_loader, device, cfg)
        lr_now = optimizer.param_groups[0]["lr"]

        # Periodically save aux + traj visualisation
        vis_every = cfg.get("vis_every_epochs", 20)
        if aux_decoder is not None and (epoch + 1) % vis_every == 0 and vis_batch is not None:
            model.eval()
            with torch.no_grad():
                cam  = vis_batch["camera"].to(device)
                hist = vis_batch["history"].to(device)
                cmd  = vis_batch["command"].to(device)
                fp, _, _ = model(cam, hist, cmd)
                traj_pred = fp[0]
                stk = model.aux_stage_tokens()
                af  = aux_decoder(stk)
                dp  = depth_head(af) if depth_head else None
                sp  = seg_head(af)   if seg_head   else None
            vpath = f"predictions_aux_epoch{epoch+1:03d}.pdf"
            visualize_aux(
                cam.cpu(),
                vis_batch.get("depth"),
                vis_batch.get("semantic_label"),
                dp.cpu() if dp is not None else None,
                sp.cpu() if sp is not None else None,
                traj_gt   = vis_batch.get("future", {}).get if False else vis_batch.get("future"),
                traj_pred = traj_pred.cpu(),
                history   = vis_batch.get("history"),
                output_path=vpath, n=4,
            )
            log.info("  Saved aux visualisation → %s", vpath)
            model.train()
        lr_now = optimizer.param_groups[0]["lr"]

        # During router warmup the router is random → val/ade is meaningless.
        # Log bofk_ade as the primary curve so W&B charts are interpretable,
        # and log router_ade separately so we can see when it converges to bofk.
        router_active = (epoch >= router_warmup_epochs) or (K == 1)
        log_metrics = {
            "train/loss":      train_loss / len(train_loader),
            "train/lr":        lr_now,
            "val/ade":         metrics["val/bofk_ade"],        # always the honest number
            "val/router_ade":  metrics["val/ade"],             # random during warmup, real after
            **{k: v for k, v in metrics.items()
               if k not in ("val/ade",)},
            "epoch":           epoch + 1,
            "router_active":   int(router_active),
        }
        wandb_log(run, global_step, log_metrics)

        log.info(
            "Epoch %3d/%d | lr %.2e | train %.5f | "
            "bofk %.5f | router_ade %.5f%s | far %.5f | %.0fs",
            epoch + 1, num_epochs, lr_now,
            train_loss / len(train_loader),
            metrics["val/bofk_ade"], metrics["val/ade"],
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
            save_checkpoint(path, model, optimizer, scheduler,
                            epoch + 1, best_ade, cfg)
            log.info("  ✓ Best ADE %.5f → %s", best_ade, path)
            best_ckpt = path
            if run:
                try:
                    run.summary["best_ade"]  = best_ade
                    run.summary["best_ckpt"] = path
                except Exception: pass

    log.info("Done. Best ADE: %.5f @ %s", best_ade, best_ckpt)
    print("Done. Best ADE: %.5f @ %s" % (best_ade, best_ckpt))
    if run:
        run.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/v3_baseline.yaml")
    parser.add_argument("--data-dir",      default=None)
    parser.add_argument("--dino-repo-dir", default=None)
    parser.add_argument("--dino-weights",  default=None)
    parser.add_argument("--no-wandb",      action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)
    log.info("Config: %s", cfg)
    print(yaml.dump(cfg, indent=4, sort_keys=False))
    train(cfg, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()