"""
Auxiliary perception heads for Phase 2: depth + semantic segmentation.

DPT-style decoder: reads patch tokens from 4 backbone stages, reassembles
to spatial maps, fuses coarse→fine via a residual feature pyramid.
Gradient is NOT detached — aux losses flow back into backbone blocks
(representation regularisation — the entire point of aux tasks).

Heads:
  DepthHead      : metric depth in metres, scale-invariant log loss
  SegHead        : per-pixel class logits, cross-entropy loss
  SurfaceNormals : derived from depth gradients, no extra labels needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from guido.dataset import GRID_H, GRID_W


# ── Feature pyramid ───────────────────────────────────────────────────────────

class Reassemble(nn.Module):
    """Flat patch tokens (B, N, d) → spatial feature map (B, d_out, H', W')."""
    def __init__(self, d_in, d_out, scale=1.0):
        super().__init__()
        self.grid_h = GRID_H
        self.grid_w = GRID_W
        self.scale  = scale
        self.proj   = nn.Conv2d(d_in, d_out, kernel_size=1)

    def forward(self, tokens):
        B, N, d = tokens.shape
        if N != self.grid_h * self.grid_w:
            raise ValueError(f"Expected {self.grid_h * self.grid_w} tokens, got {N}")
        x = tokens.permute(0, 2, 1).reshape(B, d, self.grid_h, self.grid_w)
        x = self.proj(x)
        if self.scale != 1.0:
            h = max(1, round(self.grid_h * self.scale))
            w = max(1, round(self.grid_w * self.scale))
            x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
        return x


class FusionBlock(nn.Module):
    """Upsample coarse to match fine, add, then refine with a residual conv."""
    def __init__(self, d):
        super().__init__()
        self.conv1 = nn.Conv2d(d, d, 3, padding=1)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1)
        self.norm  = nn.GroupNorm(8, d)

    def forward(self, fine, coarse):
        up = F.interpolate(coarse, fine.shape[-2:], mode="bilinear", align_corners=False)
        x  = F.relu(self.norm(self.conv1(fine + up)), inplace=True)
        return fine + self.conv2(x)   # residual


class AuxDecoder(nn.Module):
    """
    Feature pyramid decoder shared between all aux heads.
    d_dec=192 gives noticeably better spatial detail than 128 at modest
    extra cost (~+0.6M params total).
    """
    def __init__(self, d_backbone, d_dec=192, out_h=200, out_w=300):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reassemble = nn.ModuleList([
            Reassemble(d_backbone, d_dec, scale=0.5),
            Reassemble(d_backbone, d_dec, scale=1.0),
            Reassemble(d_backbone, d_dec, scale=2.0),
            Reassemble(d_backbone, d_dec, scale=4.0),
        ])
        self.fuse = nn.ModuleList([FusionBlock(d_dec) for _ in range(3)])
        self.final = nn.Sequential(
            nn.Conv2d(d_dec, d_dec, 3, padding=1),
            nn.GroupNorm(8, d_dec),
            nn.ReLU(inplace=True),
        )

    def forward(self, stage_tokens):
        maps = [r(t) for r, t in zip(self.reassemble, stage_tokens)]
        x = self.fuse[0](maps[1], maps[0])
        x = self.fuse[1](maps[2], x)
        x = self.fuse[2](maps[3], x)
        x = F.interpolate(x, (self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return self.final(x)


# ── Task heads ────────────────────────────────────────────────────────────────

class DepthHead(nn.Module):
    """
    Metric depth in metres. Two-conv refinement head followed by Softplus
    to ensure positive predictions. Trained with scale-invariant log loss.
    """
    def __init__(self, d_dec):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_dec, d_dec // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_dec // 2, d_dec // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_dec // 4, 1, 1),
            nn.Softplus(),
        )
    def forward(self, x): return self.head(x)


class SegHead(nn.Module):
    def __init__(self, d_dec, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_dec, d_dec // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_dec // 2, num_classes, 1),
        )
    def forward(self, x): return self.head(x)


# ── Losses ────────────────────────────────────────────────────────────────────

def depth_loss(pred, target, var_weight=0.85):
    """
    Scale-invariant log loss (SILog). Better than L1 for depth because
    it's robust to global scale offsets — the model learns relative depth
    structure even if the absolute scale estimate drifts.
    Only computed on valid (positive) target pixels.
    """
    valid = target > 0.1
    if not valid.any():
        return pred.sum() * 0.0
    p = torch.log(pred[valid].clamp(min=1e-3))
    t = torch.log(target[valid].clamp(min=1e-3))
    d = p - t
    return d.pow(2).mean() - var_weight * d.mean().pow(2)


def seg_loss(pred_logits, target, ignore_index=255):
    return F.cross_entropy(pred_logits, target.long(), ignore_index=ignore_index)


def normals_from_depth(depth):
    """
    Derive surface normals from a depth map via finite differences.
    Returns (B, 3, H, W), L2-normalised. Free labels from depth GT.
    Useful as an additional aux signal when depth alone converges quickly.
    """
    dz_dy = F.pad(depth[:, :, 2:] - depth[:, :, :-2], (0, 0, 1, 1))
    dz_dx = F.pad(depth[:, :, :, 2:] - depth[:, :, :, :-2], (1, 1, 0, 0))
    ones  = torch.ones_like(dz_dx)
    # Normal vector for a surface z=f(x,y): N = (-dz/dx, -dz/dy, 1), normalised
    n     = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
    return F.normalize(n, dim=1)


def normals_loss(pred_depth, target_depth):
    """L1 loss between normals derived from predicted and GT depth."""
    pred_n   = normals_from_depth(pred_depth)
    target_n = normals_from_depth(target_depth)
    valid    = (target_depth > 0.1).expand_as(target_n)
    if not valid.any():
        return pred_depth.sum() * 0.0
    return F.l1_loss(pred_n[valid], target_n[valid])


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_aux(camera, depth_gt, seg_gt, depth_pred, seg_pred,
                  traj_gt=None, traj_pred=None, history=None,
                  output_path="aux_predictions.pdf", n=4,
                  imagenet_mean=(0.485, 0.456, 0.406),
                  imagenet_std=(0.229, 0.224, 0.225)):
    """
    Save a PDF with camera, depth (GT vs pred), segmentation (GT vs pred),
    and optionally trajectory overlays for n samples.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    n     = min(n, camera.size(0))
    mean  = torch.tensor(imagenet_mean).view(3, 1, 1)
    std   = torch.tensor(imagenet_std).view(3, 1, 1)

    has_depth = depth_gt is not None and depth_pred is not None
    has_seg   = seg_gt   is not None and seg_pred   is not None
    has_traj  = traj_gt  is not None and traj_pred  is not None

    # Row layout: camera | depth_gt | depth_pred | seg_gt | seg_pred | traj
    col_labels = ["Camera"]
    if has_depth: col_labels += ["Depth GT", "Depth Pred"]
    if has_seg:   col_labels += ["Seg GT",   "Seg Pred"]
    if has_traj:  col_labels += ["Trajectory"]
    ncols = len(col_labels)

    fig, axes = plt.subplots(n, ncols, figsize=(3.5 * ncols, 3 * n),
                             squeeze=False)
    fig.suptitle("Auxiliary Predictions", fontsize=11)

    for i in range(n):
        col = 0
        # Camera
        img = (camera[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, col].imshow(img); axes[i, col].axis("off")
        if i == 0: axes[i, col].set_title(col_labels[col], fontsize=8)
        col += 1

        if has_depth:
            d_gt = depth_gt[i, 0].cpu().numpy()
            d_pr = depth_pred[i, 0].detach().cpu().numpy()
            vmin, vmax = d_gt.min(), d_gt.max()
            axes[i, col].imshow(d_gt, cmap="magma", vmin=vmin, vmax=vmax)
            axes[i, col].axis("off")
            if i == 0: axes[i, col].set_title(col_labels[col], fontsize=8)
            col += 1
            axes[i, col].imshow(d_pr, cmap="magma", vmin=vmin, vmax=vmax)
            axes[i, col].axis("off")
            if i == 0: axes[i, col].set_title(col_labels[col], fontsize=8)
            col += 1

        if has_seg:
            s_gt = seg_gt[i].cpu().numpy()
            s_pr = seg_pred[i].argmax(0).detach().cpu().numpy()
            axes[i, col].imshow(s_gt, cmap="tab20", vmin=0, vmax=13)
            axes[i, col].axis("off")
            if i == 0: axes[i, col].set_title(col_labels[col], fontsize=8)
            col += 1
            axes[i, col].imshow(s_pr, cmap="tab20", vmin=0, vmax=13)
            axes[i, col].axis("off")
            if i == 0: axes[i, col].set_title(col_labels[col], fontsize=8)
            col += 1

        if has_traj:
            ax = axes[i, col]
            if history is not None:
                h = history[i].cpu().numpy()
                ax.plot(h[:, 0], h[:, 1], "o-", c="gold", ms=3, lw=1, label="history")
            assert traj_gt is not None and traj_pred is not None
            gt = traj_gt[i].cpu().numpy()
            pr = traj_pred[i].detach().cpu().numpy()

            def plot_faded_traj(points, start_color, end_color, label):
                if len(points) < 2:
                    return
                cmap = LinearSegmentedColormap.from_list(f"{label}_fade", [start_color, end_color])
                segments = [points[j : j + 2, :2] for j in range(len(points) - 1)]
                colors = cmap(np.linspace(0.0, 1.0, len(segments)))
                ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.5, zorder=2))
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=np.linspace(0.0, 1.0, len(points)),
                    cmap=cmap,
                    s=18,
                    edgecolors="none",
                    zorder=3,
                    label=label,
                )

            plot_faded_traj(gt, "green", "turquoise", "GT")
            plot_faded_traj(pr, "red", "purple", "pred")

            # Plot last points of GT & pred as bigger, high-contrast markers.
            ax.scatter(gt[-1, 0], gt[-1, 1], s=80, c=["green"], edgecolors="white", linewidths=1.2, zorder=4)
            ax.scatter(pr[-1, 0], pr[-1, 1], s=80, c=["purple"], edgecolors="white", linewidths=1.2, zorder=4)

            all_points = [gt[:, :2], pr[:, :2]]
            if history is not None:
                all_points.append(history[i].cpu().numpy()[:, :2])
            pts = np.concatenate(all_points, axis=0)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            x_pad = max(0.5, 0.15 * (x_max - x_min))
            y_pad = max(0.5, 0.15 * (y_max - y_min))
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect("equal")
            # ax.axis("off")
            if i == 0:
                ax.set_title(col_labels[col], fontsize=8)
                ax.legend(fontsize=6, loc="upper left")
            col += 1

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    return output_path
