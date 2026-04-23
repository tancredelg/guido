"""
DrivingPlannerV2 — Coarse-to-Fine with trajectory-conditioned routing
=====================================================================

Key insight from experiments
-----------------------------
The original router failed because it predicted "which head wins" from the
same input features the heads use to produce trajectories. The winner depends
on trajectory quality (ADE vs GT), not just input features — so the router
had no privileged information to make that decision.

Fix: coarse-to-fine decoding. Each head first produces a cheap coarse
prediction (a few anchor waypoints), then the router sees ALL coarse
predictions simultaneously and picks which head's trajectory to refine.
This gives the router actual trajectory content to compare — not just input
features — making it a genuine quality discriminator.

Information flow
----------------
camera → DINOv3 ViT-B → cls (B,D), patches (B,256,D)
history → TransformerEncoder → hist_tokens (B,21,d), hist_cls (B,d)
command → Embedding → cmd (B,d)

SceneMotionFusion: hist_tokens attend over patches → enriched_hist (B,21,d)
ctx = proj(cat[img_cls, hist_cls, cmd])  → (B,d)

Stage 1 — Coarse heads (lightweight, shared architecture):
    Each head: MLP(fused) → (B, N_coarse, 2)  e.g. 6 anchor waypoints
    These are cheap and provide trajectory-level content to the router.

Router (trajectory-conditioned):
    Input: cat([ctx, coarse_0, ..., coarse_{K-1}])  all flattened
    → MLP → (B, K) logits
    During warmup: router frozen, but coarse heads still train
    After warmup: router sees meaningful coarse predictions to compare

Stage 2 — Fine heads (transformer decoder, same as before):
    Each head: 60 learned queries attend over enriched context
    → (B, 60, 2) full trajectory

Training: WTA on fine predictions, router CE on coarse-informed logits
Inference: router argmax selects the fine head
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _proj(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


def _sinusoidal_pe(length: int, d: int) -> torch.Tensor:
    pe = torch.zeros(length, d)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# History Transformer encoder
# ─────────────────────────────────────────────────────────────────────────────


class HistoryEncoder(nn.Module):
    def __init__(self, in_dim=4, d=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        self.register_buffer("pe", _sinusoidal_pe(21, d))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, history):
        x = self.embed(history) + self.pe.unsqueeze(0)
        x = self.norm(self.encoder(x))
        return x, x.mean(dim=1)  # tokens (B,21,d), cls (B,d)


# ─────────────────────────────────────────────────────────────────────────────
# Scene-motion cross-attention
# ─────────────────────────────────────────────────────────────────────────────


class SceneMotionFusion(nn.Module):
    """History tokens attend over image patch tokens."""

    def __init__(self, d, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, hist_tokens, patch_tokens):
        q = self.norm_q(hist_tokens)
        kv = self.norm_kv(patch_tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(hist_tokens + out)


# ─────────────────────────────────────────────────────────────────────────────
# Coarse head (Stage 1) — cheap MLP, produces anchor waypoints
# ─────────────────────────────────────────────────────────────────────────────


class CoarseHead(nn.Module):
    """
    Lightweight MLP producing N_coarse anchor waypoints from the fused vector.
    These are used by the router to compare trajectory shapes across heads.
    They are also used as auxiliary supervision targets (interpolated from GT).
    """

    def __init__(self, fused_dim: int, d: int, n_coarse: int = 6, dropout: float = 0.05):
        super().__init__()
        self.n_coarse = n_coarse
        self.net = nn.Sequential(
            nn.Linear(fused_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, n_coarse * 2),
        )

    def forward(self, fused):
        return self.net(fused).reshape(fused.size(0), self.n_coarse, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Fine head (Stage 2) — transformer decoder, full 60-step trajectory
# ─────────────────────────────────────────────────────────────────────────────


class FineHead(nn.Module):
    def __init__(self, d, num_heads=4, num_layers=2, dropout=0.1, num_waypoints=60):
        super().__init__()
        self.T = num_waypoints
        self.query_embed = nn.Embedding(num_waypoints, d)
        self.register_buffer("pe", _sinusoidal_pe(num_waypoints, d))
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d, 2)

    def forward(self, hist_tokens, ctx_token):
        B = hist_tokens.size(0)
        context = torch.cat([hist_tokens, ctx_token], dim=1)  # (B, 22, d)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) + self.pe.unsqueeze(0)
        return self.out_proj(self.decoder(queries, context))  # (B, 60, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main V2 model
# ─────────────────────────────────────────────────────────────────────────────


class DrivingPlannerV2(nn.Module):
    """
    Config keys
    -----------
    d                : inner model dim (default 256)
    K                : number of mixture heads (default 3; 1 = single head, no routing)
    n_coarse         : anchor waypoints per coarse head (default 6)
    unfreeze_blocks  : unfreeze last N ViT-B blocks (default 1)
    hist_layers      : transformer encoder layers (default 2)
    dec_layers       : fine decoder layers per head (default 2)
    num_heads        : attention heads (default 4)
    dropout          : dropout (default 0.05)
    cmd_embed_dim    : command embedding dim (default 64)
    smoothness_weight: smoothness regulariser weight (default 0.05)
    """

    def __init__(
        self,
        *,
        dino_model: str = "dinov3_vitb16",
        dino_repo_dir: str = "",
        dino_weights: str = "",
        unfreeze_blocks: int = 1,
        d: int = 256,
        num_heads: int = 4,
        hist_layers: int = 2,
        dec_layers: int = 2,
        K: int = 3,
        n_coarse: int = 6,
        dropout: float = 0.05,
        smoothness_weight: float = 0.05,
        num_waypoints: int = 60,
        cmd_embed_dim: int = 64,
    ):
        super().__init__()
        self.K = K
        self.n_coarse = n_coarse
        self.num_waypoints = num_waypoints
        self.smoothness_weight = smoothness_weight
        self.unfreeze_blocks = unfreeze_blocks

        # ── Backbone ──────────────────────────────────────────────────────
        if not dino_repo_dir or not dino_weights:
            raise ValueError("dino_repo_dir and dino_weights must be set.")
        self.backbone = torch.hub.load(
            dino_repo_dir,
            dino_model,
            source="local",
            weights=dino_weights,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0 and hasattr(self.backbone, "blocks"):
            for blk in self.backbone.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        dino_dim: int = self.backbone.embed_dim

        # ── History encoder ───────────────────────────────────────────────
        self.hist_enc = HistoryEncoder(4, d, num_heads, hist_layers, dropout)
        self.cmd_embed = nn.Embedding(3, cmd_embed_dim)
        self.img_proj = _proj(dino_dim, d)
        self.patch_proj = nn.Linear(dino_dim, d)
        self.cmd_proj = _proj(cmd_embed_dim, d)
        self.scene_motion = SceneMotionFusion(d, num_heads, dropout)
        self.ctx_proj = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        fused_dim = d * 3  # img_cls | hist_cls | cmd

        # ── Stage 1: coarse heads ─────────────────────────────────────────
        # Always built even for K=1 (used as auxiliary loss / better gradient)
        self.coarse_heads = nn.ModuleList(
            [CoarseHead(fused_dim, d, n_coarse, dropout) for _ in range(max(K, 1))]
        )

        # ── Stage 2: fine heads ───────────────────────────────────────────
        self.fine_heads = nn.ModuleList(
            [FineHead(d, num_heads, dec_layers, dropout, num_waypoints) for _ in range(max(K, 1))]
        )

        # ── Router (trajectory-conditioned) ───────────────────────────────
        # Input: ctx vector + all K coarse predictions (flattened)
        # This gives the router actual trajectory shapes to compare.
        if K > 1:
            router_in = d + K * n_coarse * 2
            self.router = nn.Sequential(
                nn.Linear(router_in, d),
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d, d // 2),
                nn.LayerNorm(d // 2),
                nn.GELU(),
                nn.Linear(d // 2, K),
            )

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────

    def _init_weights(self):
        own = [self.img_proj, self.patch_proj, self.cmd_proj, self.ctx_proj, self.scene_motion]
        if self.K > 1:
            own.append(self.router)
        for module in own:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for head in self.fine_heads:
            nn.init.xavier_uniform_(head.out_proj.weight, gain=0.1)
            nn.init.zeros_(head.out_proj.bias)
        for head in self.coarse_heads:
            # small init so coarse predictions start near zero
            nn.init.xavier_uniform_(head.net[-1].weight, gain=0.1)
            nn.init.zeros_(head.net[-1].bias)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Backbone ──────────────────────────────────────────────────────────

    def _backbone_features(self, camera):
        feats = self.backbone.forward_features(camera)
        return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, camera, history, command):
        B = camera.size(0)

        # 1. Image
        if self.unfreeze_blocks == 0:
            with torch.no_grad():
                cls, patches = self._backbone_features(camera)
        else:
            cls, patches = self._backbone_features(camera)

        # 2. History
        hist_tokens, hist_cls = self.hist_enc(history)
        cmd = self.cmd_proj(self.cmd_embed(command))
        img_cls = self.img_proj(cls)
        patch_proj = self.patch_proj(patches)
        hist_tokens = self.scene_motion(hist_tokens, patch_proj)

        # 3. Global context
        ctx = self.ctx_proj(torch.cat([img_cls, hist_cls, cmd], dim=-1))  # (B,d)
        ctx_token = ctx.unsqueeze(1)  # (B,1,d)

        # 4. Fused vector for coarse heads and concat baseline
        fused = torch.cat([img_cls, hist_cls, cmd], dim=-1)  # (B, d*3)

        # 5. Stage 1: coarse predictions from each head
        coarse_preds = [h(fused) for h in self.coarse_heads]  # K × (B, n_coarse, 2)

        # 6. Router: sees ctx + all coarse predictions flattened
        if self.K > 1:
            coarse_flat = torch.cat([c.reshape(B, -1) for c in coarse_preds], dim=-1)  # (B, K*n_coarse*2)
            router_in = torch.cat([ctx, coarse_flat], dim=-1)  # (B, d + K*n_coarse*2)
            router_logits = self.router(router_in)  # (B, K)
        else:
            router_logits = torch.zeros(B, 1, device=camera.device)

        # 7. Stage 2: fine predictions from each head
        fine_preds = [h(hist_tokens, ctx_token) for h in self.fine_heads]  # K × (B, 60, 2)

        return fine_preds, coarse_preds, router_logits

    # ── Utilities ─────────────────────────────────────────────────────────

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())
