"""
DrivingPlannerV2 — separate CoarseHead MLP + full-capacity FineHead decoder.

Architecture
------------
camera  → DINOv3 ViT-B → cls (B,D), patches (B,256,D)
history → TransformerEncoder → hist_tokens (B,21,d), hist_cls (B,d)
command → Embedding → cmd (B,d)

SceneMotionFusion: hist_tokens attend over patches → enriched hist_tokens
ctx = proj(cat[img_cls, hist_cls, cmd])  (B,d)
fused = cat[img_cls, hist_cls, cmd]      (B,d*3)  — for coarse MLP inputs

Per head:
  CoarseHead (MLP): fused → (B, n_coarse, 2)   separate, keeps fine capacity
  FineHead  (TFmr): hist_tokens + ctx → (B, 60, 2)  all dec_layers available

Router (K>1, n_coarse>0): cat[ctx, all_coarse_flat] → MLP → (B,K) logits
  Trajectory-conditioned: router sees actual trajectory sketches to compare.

n_coarse=0 → CoarseHead disabled, router sees ctx only (or no router for K=1)
K=1        → no router, no WTA, just single FineHead + optional coarse aux loss
"""

import math
import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ── Utilities ─────────────────────────────────────────────────────────────────


def _proj(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())


def _sinusoidal_pe(length, d):
    pe = torch.zeros(length, d)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


# ── History encoder ───────────────────────────────────────────────────────────


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


# ── Scene-motion cross-attention ──────────────────────────────────────────────


class SceneMotionFusion(nn.Module):
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


# ── Coarse head (separate MLP) ────────────────────────────────────────────────


class CoarseHead(nn.Module):
    """
    Lightweight MLP: fused_vector → n_coarse anchor waypoints.
    Completely separate from the fine decoder — preserves its full capacity.
    Used for (a) router trajectory conditioning and (b) auxiliary coarse loss.
    """

    def __init__(self, fused_dim, d, n_coarse, dropout=0.05):
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


# ── Fine head (full-capacity transformer decoder) ─────────────────────────────


class FineHead(nn.Module):
    """
    All dec_layers used for fine trajectory decoding.
    60 learned queries + sinusoidal PE cross-attend over
    [hist_tokens (B,21,d) | ctx_token (B,1,d)].
    """

    def __init__(self, d, num_heads=4, dec_layers=3, dropout=0.1, num_waypoints=60):
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
        self.decoder = nn.TransformerDecoder(layer, num_layers=dec_layers)
        self.fine_proj = nn.Linear(d, 2)

    def forward(self, hist_tokens, ctx_token):
        B = hist_tokens.size(0)
        context = torch.cat([hist_tokens, ctx_token], dim=1)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) + self.pe.unsqueeze(0)
        return self.fine_proj(self.decoder(queries, context))  # (B, 60, 2)


# ── Main model ────────────────────────────────────────────────────────────────


class DrivingPlannerV2(nn.Module):
    """
    Config keys
    -----------
    d               : inner dim (default 256)
    K               : mixture heads; 1 = no routing
    n_coarse        : anchor waypoints for coarse MLP; 0 = disabled
    unfreeze_blocks : unfreeze last N ViT-B blocks
    hist_layers     : history transformer encoder layers
    dec_layers      : fine decoder layers per head
    num_heads       : attention heads
    dropout, cmd_embed_dim, smoothness_weight : as named
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
        dec_layers: int = 3,
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
        dino_dim = self.backbone.embed_dim

        # ── Encoders ──────────────────────────────────────────────────────
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
        fused_dim = d * 3

        # ── Coarse heads (separate MLP, optional) ─────────────────────────
        self.coarse_heads = (
            nn.ModuleList([CoarseHead(fused_dim, d, n_coarse, dropout) for _ in range(max(K, 1))])
            if n_coarse > 0
            else None
        )

        # ── Fine heads (full-capacity transformer decoder) ─────────────────
        self.heads = nn.ModuleList(
            [FineHead(d, num_heads, dec_layers, dropout, num_waypoints) for _ in range(max(K, 1))]
        )

        # ── Router (trajectory-conditioned when n_coarse > 0) ─────────────
        if K > 1:
            router_in = d + K * n_coarse * 2 if n_coarse > 0 else d
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
        if self.coarse_heads is not None:
            own.extend(self.coarse_heads)
        for module in own:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for head in self.heads:
            nn.init.xavier_uniform_(head.fine_proj.weight, gain=0.1)
            nn.init.zeros_(head.fine_proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Backbone features ─────────────────────────────────────────────────

    def _backbone_features(self, camera):
        feats = self.backbone.forward_features(camera)
        return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, camera, history, command):
        B = camera.size(0)

        # Image
        if self.unfreeze_blocks == 0:
            with torch.no_grad():
                cls, patches = self._backbone_features(camera)
        else:
            cls, patches = self._backbone_features(camera)

        # History + command
        hist_tokens, hist_cls = self.hist_enc(history)
        cmd = self.cmd_proj(self.cmd_embed(command))
        img_cls = self.img_proj(cls)
        hist_tokens = self.scene_motion(hist_tokens, self.patch_proj(patches))

        # Context
        ctx = self.ctx_proj(torch.cat([img_cls, hist_cls, cmd], dim=-1))
        ctx_token = ctx.unsqueeze(1)
        fused = torch.cat([img_cls, hist_cls, cmd], dim=-1)

        # Coarse predictions (separate MLP)
        if self.coarse_heads is not None:
            coarse_preds = [h(fused) for h in self.coarse_heads]
        else:
            coarse_preds = [None] * max(self.K, 1)

        # Fine predictions (full transformer decoder)
        fine_preds = [h(hist_tokens, ctx_token) for h in self.heads]

        # Router
        if self.K > 1 and self.n_coarse > 0:
            coarse_flat = torch.cat([c.reshape(B, -1) for c in coarse_preds], dim=-1)
            router_logits = self.router(torch.cat([ctx, coarse_flat], dim=-1))
        elif self.K > 1:
            router_logits = self.router(ctx)
        else:
            router_logits = torch.zeros(B, 1, device=camera.device)

        return fine_preds, coarse_preds, router_logits

    # ── Utilities ─────────────────────────────────────────────────────────

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())
