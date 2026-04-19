"""
DrivingPlanner — configurable backbone + fusion + decoder.

Backbone  : any DINOv3 ViT or ConvNeXt (auto-detects embed_dim)
Fusion    : fusion_arch = "concat" | "crossattn"
Decoder   : decoder_arch = "mlp" | "conv" | "transformer"

The three decoder options address the far-horizon ADE collapse:
  mlp         — baseline flat MLP, no temporal structure (ade_far ~4.0)
  conv        — 12 sparse anchors → transposed-conv upsample → 60 steps.
                Enforces smoothness by construction; no training loop changes.
  transformer — 60 learned timestep queries cross-attend over the full 21-step
                history sequence + image/command context.  Each output step
                explicitly sees the car's motion history, giving strong temporal
                coherence at long horizon.
"""

import torch
import torch.nn as nn
import math
import logging

log = logging.getLogger(__name__)


# ── Backbone helpers ───────────────────────────────────────────────────────────


def _is_convnext(name: str) -> bool:
    return "convnext" in name.lower()


def _backbone_features(backbone, camera: torch.Tensor):
    """
    Returns (cls_token, patch_tokens) from any DINOv3 backbone.
    cls_token   : (B, D)    — global summary
    patch_tokens: (B, N, D) — spatial tokens (None for ConvNeXt)
    """
    feats = backbone.forward_features(camera)
    if isinstance(feats, dict):
        cls = feats.get("x_norm_clstoken", None)
        patches = feats.get("x_norm_patchtokens", None)
        if cls is None:
            # ConvNeXt: spatial map → GAP
            spatial = feats.get("x_norm_patchtokens", feats.get("x", None))
            if spatial is not None:
                cls = spatial.mean(dim=1) if spatial.dim() == 3 else spatial.mean(dim=[2, 3])
        return cls, patches
    # Fallback: tensor output → GAP
    return feats.mean(dim=list(range(2, feats.dim()))), None


# ── Cross-attention fusion (optional) ─────────────────────────────────────────


class CrossAttnFusion(nn.Module):
    """Motion query attends over DINO spatial patch tokens."""

    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, query, context):
        q = self.norm_q(query).unsqueeze(1)
        kv = self.norm_kv(context)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(out.squeeze(1) + query)


# ── Decoders ───────────────────────────────────────────────────────────────────


class MLPDecoder(nn.Module):
    """Baseline: flat MLP from fused vector → 60×2.  No temporal structure."""

    def __init__(self, in_dim: int, fusion_dim: int, num_waypoints: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_waypoints * 2),
        )
        self.T = num_waypoints

    def forward(self, fused, hist_seq=None):
        return self.net(fused).reshape(fused.size(0), self.T, 2)


class ConvDecoder(nn.Module):
    """
    Predict N_anchors sparse waypoints via MLP, then upsample to T=60 via
    transposed convolution.

    Smoothness is enforced structurally: the transposed conv must produce a
    continuous interpolation between anchor points, so adjacent timesteps are
    physically related.  This directly attacks far-horizon ADE collapse.

    Architecture:
        fused → Linear → (B, N_anchors * 2)
             → reshape → (B, 2, N_anchors)
             → ConvTranspose1d(2, 32, kernel=5, stride=5)  [N→5N]
             → GELU
             → Conv1d(32, 2, kernel=3, padding=1)           [refine]
             → (B, 2, T) → permute → (B, T, 2)

    With N_anchors=12, stride=5: output = (12-1)*5 + 5 = 60 ✓
    """

    def __init__(
        self,
        in_dim: int,
        fusion_dim: int,
        num_waypoints: int = 60,
        n_anchors: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_waypoints == n_anchors * 5, (
            f"num_waypoints ({num_waypoints}) must equal n_anchors*5 ({n_anchors * 5})"
        )
        self.T = num_waypoints
        self.n_anchors = n_anchors

        self.anchor_head = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_anchors * 2),
        )
        # upsample n_anchors → 60 via transposed conv
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(2, 32, kernel_size=5, stride=5),  # (B, 32, 60)
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),  # (B, 2, 60)
        )

    def forward(self, fused, hist_seq=None):
        B = fused.size(0)
        anchors = self.anchor_head(fused)  # (B, n_anchors*2)
        anchors = anchors.reshape(B, 2, self.n_anchors)  # (B, 2, n_anchors)
        out = self.upsample(anchors)  # (B, 2, 60)
        return out.permute(0, 2, 1)  # (B, 60, 2)


class TransformerDecoder(nn.Module):
    """
    60 learned timestep queries cross-attend over the full 21-step history
    sequence concatenated with the image/command context vector.

    Why this helps far-horizon ADE: each output timestep explicitly attends
    to the car's motion history and can learn "at t=50, I should be consistent
    with the turning rate I saw in the history", rather than extrapolating
    blindly from a single fused vector.

    Architecture:
        context = cat([history_sequence (B,21,D), ctx_token (B,1,D)], dim=1)
        queries  = learned_embed (B, 60, D) + sinusoidal pos encoding
        output   = MultiheadAttention(queries, context, context)
                 → LayerNorm → FFN → (B, 60, 2)
    """

    def __init__(
        self,
        in_dim: int,
        hist_hidden_dim: int,
        num_waypoints: int = 60,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        patch_dim: int = 0,  # if >0, add a patch_proj and accept patch tokens
    ):
        super().__init__()
        self.T = num_waypoints
        self.d_model = d_model

        self.ctx_proj = nn.Linear(in_dim, d_model)
        self.hist_proj = nn.Linear(hist_hidden_dim, d_model)
        if patch_dim > 0:
            self.patch_proj = nn.Linear(patch_dim, d_model)

        self.query_embed = nn.Embedding(num_waypoints, d_model)
        pe = self._make_sinusoidal_pe(num_waypoints, d_model)
        self.register_buffer("pe", pe)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 2)

    @staticmethod
    def _make_sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        return pe

    def forward(self, fused, hist_seq, patch_tokens=None):
        """
        fused        : (B, in_dim)          — image CLS + command context
        hist_seq     : (B, 21, hist_hidden) — full GRU output sequence
        patch_tokens : (B, N, dino_dim) optional — spatial patch tokens
                       from backbone, projected and appended to context so
                       decoder queries can attend to road geometry directly.
        """
        B = fused.size(0)

        hist_ctx = self.hist_proj(hist_seq)  # (B, 21, d_model)
        ctx_tok = self.ctx_proj(fused).unsqueeze(1)  # (B,  1, d_model)
        context = torch.cat([hist_ctx, ctx_tok], dim=1)  # (B, 22, d_model)

        if patch_tokens is not None and hasattr(self, "patch_proj"):
            patches_proj = self.patch_proj(patch_tokens)  # (B, N, d_model)
            context = torch.cat([context, patches_proj], dim=1)  # (B, 22+N, d_model)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.pe.unsqueeze(0)

        out = self.decoder(queries, context)
        return self.out_proj(out)


# ── Main model ─────────────────────────────────────────────────────────────────


class DrivingPlanner(nn.Module):
    """
    Config keys
    -----------
    fusion_arch   : "concat"      | "crossattn"
    decoder_arch  : "mlp"         | "conv"        | "transformer"
    decoder_d     : inner dim for transformer decoder (default 128)
    decoder_layers: number of TransformerDecoderLayer (default 2)
    n_anchors     : anchor waypoints for conv decoder (default 12, must give T=60)
    """

    def __init__(
        self,
        *,
        dino_model: str = "dinov3_vitb16",
        dino_repo_dir: str = "",
        dino_weights: str = "",
        hist_input_dim: int = 4,
        hist_hidden_dim: int = 128,
        hist_num_layers: int = 2,
        cmd_embed_dim: int = 32,
        fusion_dim: int = 256,
        num_heads: int = 4,
        num_waypoints: int = 60,
        dropout: float = 0.05,
        fusion_arch: str = "concat",  # "concat" | "crossattn"
        decoder_arch: str = "mlp",  # "mlp" | "conv" | "transformer"
        decoder_d: int = 128,  # transformer decoder inner dim
        decoder_layers: int = 2,
        n_anchors: int = 12,
        unfreeze_blocks: int = 0,
        residual_baseline: bool = False,
        decoder_patches: bool = False,  # give transformer decoder direct patch token access
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.fusion_arch = fusion_arch
        self.decoder_arch = decoder_arch
        self.unfreeze_blocks = unfreeze_blocks
        self.residual_baseline = residual_baseline
        self.decoder_patches = decoder_patches
        cfg_decoder_patches = decoder_patches

        if not dino_repo_dir or not dino_weights:
            raise ValueError("Both dino_repo_dir and dino_weights must be set.")

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = torch.hub.load(
            dino_repo_dir,
            dino_model,
            source="local",
            weights=dino_weights,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally unfreeze the last N transformer blocks for fine-tuning.
        # Use a much lower lr for these params (see train.py param groups).
        # Only meaningful for ViT backbones (ignored for ConvNeXt).
        if unfreeze_blocks > 0 and hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            log.info(
                "Unfreezing last %d backbone blocks (%d params)",
                unfreeze_blocks,
                sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            )

        dino_dim: int = self.backbone.embed_dim

        # ── History encoder ───────────────────────────────────────────────────
        # Always return full sequence (needed by transformer decoder).
        self.history_gru = nn.GRU(
            input_size=hist_input_dim,
            hidden_size=hist_hidden_dim,
            num_layers=hist_num_layers,
            batch_first=True,
            dropout=dropout if hist_num_layers > 1 else 0.0,
        )
        self.cmd_embed = nn.Embedding(3, cmd_embed_dim)

        # ── Projection heads ──────────────────────────────────────────────────
        def _proj(d):
            return nn.Sequential(nn.Linear(d, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        self.img_proj = _proj(dino_dim)
        self.hist_proj = _proj(hist_hidden_dim)
        self.cmd_proj = _proj(cmd_embed_dim)

        # ── Fusion ────────────────────────────────────────────────────────────
        if fusion_arch == "crossattn":
            self.patch_proj = nn.Linear(dino_dim, fusion_dim)
            self.motion_proj = nn.Sequential(
                nn.Linear(hist_hidden_dim + cmd_embed_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
            self.cross_attn = CrossAttnFusion(fusion_dim, num_heads, dropout)
            fused_dim = fusion_dim * 2
        else:
            fused_dim = fusion_dim * 3  # img | hist | cmd

        # ── Decoder ───────────────────────────────────────────────────────────
        if decoder_arch == "conv":
            self.decoder = ConvDecoder(
                fused_dim,
                fusion_dim,
                num_waypoints,
                n_anchors,
                dropout,
            )
        elif decoder_arch == "transformer":
            # When decoder_patches=True, pass dino_dim so the decoder can
            # project and attend over spatial patch tokens directly.
            pdim = dino_dim if cfg_decoder_patches else 0
            self.decoder = TransformerDecoder(
                fused_dim,
                hist_hidden_dim,
                num_waypoints,
                d_model=decoder_d,
                num_heads=num_heads,
                num_layers=decoder_layers,
                dropout=dropout,
                patch_dim=pdim,
            )
        else:  # mlp
            self.decoder = MLPDecoder(fused_dim, fusion_dim, num_waypoints, dropout)

        self._init_weights()

    def _physics_baseline(self, history: torch.Tensor) -> torch.Tensor:
        """
        Constant-velocity baseline: extrapolate the last observed velocity
        (pos[-1] - pos[-2]) forward for T steps.

        Returns (B, T, 2) absolute positions.
        This is subtracted from the target during training and added back at
        inference, so the model only needs to learn the residual deviation.
        """
        # history: (B, 21, 4)  cols: [x, y, sin_h, cos_h]
        last_vel = history[:, -1, :2] - history[:, -2, :2]
        steps = torch.arange(1, self.num_waypoints + 1, device=history.device).float()
        return last_vel.unsqueeze(1) * steps.unsqueeze(0).unsqueeze(-1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, p in self.history_gru.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_uniform_(p, nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, camera, history, command):
        # 1. Backbone features.
        # Only wrap in no_grad when the backbone is fully frozen — if any blocks
        # are unfrozen (unfreeze_blocks > 0), we need gradients to flow through
        # them. torch.no_grad() silently kills all gradients regardless of
        # requires_grad, which is why 1-block and 2-block unfreezing produced
        # identical loss curves: the unfrozen blocks were never actually trained.
        backbone_frozen = self.unfreeze_blocks == 0
        if backbone_frozen:
            with torch.no_grad():
                cls, patches = _backbone_features(self.backbone, camera)
        else:
            cls, patches = _backbone_features(self.backbone, camera)

        # 2. History: full sequence + last hidden state
        hist_seq, h_n = self.history_gru(history)
        hist_last = h_n[-1]
        cmd_feat = self.cmd_embed(command)

        # 3. Fusion
        if self.fusion_arch == "crossattn":
            motion_q = self.motion_proj(torch.cat([hist_last, cmd_feat], dim=-1))
            img_ctx = self.cross_attn(motion_q, self.patch_proj(patches))
            fused = torch.cat([img_ctx, motion_q], dim=-1)
        else:
            fused = torch.cat(
                [
                    self.img_proj(cls),
                    self.hist_proj(hist_last),
                    self.cmd_proj(cmd_feat),
                ],
                dim=-1,
            )

        # 4. Decode — pass patch tokens to transformer decoder if requested
        patch_ctx = patches if (self.decoder_patches and patches is not None) else None
        residual = self.decoder(fused, hist_seq, patch_ctx)

        # 5. Add constant-velocity baseline if enabled
        if self.residual_baseline:
            return self._physics_baseline(history) + residual
        return residual

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        """Unfrozen backbone params — need a much lower lr than the heads."""
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def head_parameters(self):
        """All trainable params that are NOT backbone — the heads."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return (p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())
