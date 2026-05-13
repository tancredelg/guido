"""
DrivingPlannerV4 — trajectory planner with optional perception aux heads.

Architecture:
  DINOv3 ViT-B/16  →  CLS token + patch tokens
  History (21 steps)  →  Transformer encoder  →  enriched tokens
  Scene-motion fusion: history tokens attend over patch tokens (RoPE cross-attn)
  Context vector: proj(img_cls | hist_cls)
  Coarse MLP head: 5 anchor waypoints (auxiliary loss)
  Fine transformer decoder: 60-step trajectory

Config flags:
  use_cmd   : include driving command embedding (False when all commands identical)
  scene_fusion: 'rope_xattn' (default) | 'none' (ablation: no image fusion)
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from guido.dataset import GRID_H, GRID_W

log = logging.getLogger(__name__)


def _proj(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())


def _sinusoidal_pe(length, d):
    pe  = torch.zeros(length, d)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d // 2])
    return pe


# ── 2D RoPE ───────────────────────────────────────────────────────────────────

def _build_2d_rope_cache(grid_h, grid_w, d_head, base=10000.0):
    assert d_head % 4 == 0
    d_quad = d_head // 4
    freqs  = 1.0 / (base ** (torch.arange(0, d_quad, dtype=torch.float32) / d_quad))
    h_ang  = torch.einsum("i,j->ij", torch.arange(grid_h, dtype=torch.float32), freqs)
    w_ang  = torch.einsum("i,j->ij", torch.arange(grid_w, dtype=torch.float32), freqs)
    h_ang  = h_ang.unsqueeze(1).expand(grid_h, grid_w, d_quad)
    w_ang  = w_ang.unsqueeze(0).expand(grid_h, grid_w, d_quad)
    angles = torch.cat([h_ang, w_ang], dim=-1)          # (H, W, d_half)
    cos    = angles.cos().repeat_interleave(2, dim=-1)   # (H, W, d_head)
    sin    = angles.sin().repeat_interleave(2, dim=-1)
    return cos.reshape(grid_h * grid_w, d_head), sin.reshape(grid_h * grid_w, d_head)


def _rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class RoPECrossAttention(nn.Module):
    """History tokens attend over patch tokens; keys rotated by 2D RoPE."""
    def __init__(self, d, num_heads, grid_h, grid_w, dropout=0.05, base=10000.0):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.d_head    = d // num_heads
        self.q_proj    = nn.Linear(d, d)
        self.k_proj    = nn.Linear(d, d)
        self.v_proj    = nn.Linear(d, d)
        self.out_proj  = nn.Linear(d, d)
        self.norm_q    = nn.LayerNorm(d)
        self.norm_kv   = nn.LayerNorm(d)
        self.norm_out  = nn.LayerNorm(d)
        self.dropout   = dropout
        cos, sin = _build_2d_rope_cache(grid_h, grid_w, self.d_head, base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, query, patches):
        B, Nq, _ = query.shape
        def split(x): return x.reshape(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        q = split(self.q_proj(self.norm_q(query)))
        k = split(self.k_proj(self.norm_kv(patches)))
        v = split(self.v_proj(patches))
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        ).transpose(1, 2).reshape(B, Nq, -1)
        return self.norm_out(query + self.out_proj(out))


# ── Encoders ──────────────────────────────────────────────────────────────────

class HistoryEncoder(nn.Module):
    def __init__(self, d, num_heads=4, num_layers=2, dropout=0.05):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.register_buffer("pe", _sinusoidal_pe(21, d))
        layer = nn.TransformerEncoderLayer(
            d, num_heads, d * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm    = nn.LayerNorm(d)

    def forward(self, history):
        x = self.embed(history) + self.pe.unsqueeze(0)
        x = self.norm(self.encoder(x))
        return x, x.mean(dim=1)


# ── Trajectory heads ──────────────────────────────────────────────────────────

class CoarseHead(nn.Module):
    def __init__(self, in_dim, d, n_coarse, dropout=0.05):
        super().__init__()
        self.n_coarse = n_coarse
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d, n_coarse * 2),
        )
    def forward(self, x):
        return self.net(x).reshape(x.size(0), self.n_coarse, 2)


class FineHead(nn.Module):
    """
    Transformer decoder for trajectory prediction.

    causal=True: self-attention between queries is causally masked so query t
    can only attend to queries 0..t-1. This forces temporal ordering and
    prevents far-horizon queries from "cheating" by seeing near-horizon outputs.
    Near-horizon performance is unaffected; far-horizon ADE improves because
    each step must be consistent with all prior steps.

    causal=False: standard non-causal self-attention (original behaviour).
    """
    def __init__(self, d, num_heads=4, num_layers=2, dropout=0.05, T=60, causal=False):
        super().__init__()
        self.causal = causal
        self.query_embed = nn.Embedding(T, d)
        self.register_buffer("pe", _sinusoidal_pe(T, d))
        layer = nn.TransformerDecoderLayer(
            d, num_heads, d * 4, dropout, batch_first=True, norm_first=True
        )
        self.decoder  = nn.TransformerDecoder(layer, num_layers)
        self.out_proj = nn.Linear(d, 2)
        if causal:
            # Upper-triangular mask: position i masked from attending to i+1..T-1
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", mask)

    def forward(self, memory, ctx_token):
        B = memory.size(0)
        context = torch.cat([memory, ctx_token], dim=1)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) + self.pe.unsqueeze(0)
        mask = self.causal_mask if self.causal else None
        return self.out_proj(self.decoder(q, context, tgt_mask=mask))


# ── Main model ────────────────────────────────────────────────────────────────

class DrivingPlannerV4(nn.Module):
    def __init__(
        self,
        *,
        dino_model: str          = "dinov3_vitb16",
        dino_repo_dir: str       = "",
        dino_weights: str        = "",
        unfreeze_blocks: int     = 1,
        d: int                   = 384,
        num_heads: int           = 4,
        hist_layers: int         = 2,
        dec_layers: int          = 3,
        causal_decoder: bool     = False,
        K: int                   = 1,
        n_coarse: int            = 6,
        dropout: float           = 0.05,
        smoothness_weight: float = 0.08,
        num_waypoints: int       = 60,
        cmd_embed_dim: int       = 64,
        use_cmd: bool            = False,
        rope_base: float         = 10000.0,
        scene_fusion: str        = "rope_xattn",  # "rope_xattn" | "none"
    ):
        super().__init__()
        self.K                 = K
        self.n_coarse          = n_coarse
        self.unfreeze_blocks   = unfreeze_blocks
        self.use_cmd           = use_cmd
        self.scene_fusion      = scene_fusion
        self.smoothness_weight = smoothness_weight

        # Backbone
        self.backbone = torch.hub.load(
            dino_repo_dir, dino_model, source="local", weights=dino_weights,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0 and hasattr(self.backbone, "blocks"):
            for blk in self.backbone.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        dino_dim = self.backbone.embed_dim
        n_blocks  = len(self.backbone.blocks)

        # Hooks for aux decoder (4 evenly-spaced blocks, last n_patches tokens)
        self._aux_feats: dict = {}
        n_patches = GRID_H * GRID_W
        hook_idxs = [
            max(0, n_blocks // 4 - 1),
            max(0, n_blocks // 2 - 1),
            max(0, 3 * n_blocks // 4 - 1),
            n_blocks - 1,
        ]
        self._aux_block_indices = hook_idxs

        def _make_hook(idx):
            def hook(module, inp, out):
                x = out[0] if isinstance(out, (list, tuple)) else out
                self._aux_feats[idx] = x[:, -n_patches:]
            return hook

        for idx in hook_idxs:
            self.backbone.blocks[idx].register_forward_hook(_make_hook(idx))

        # Image & history encoders
        self.img_proj    = _proj(dino_dim, d)
        self.patch_proj  = nn.Linear(dino_dim, d)
        self.hist_enc    = HistoryEncoder(d, num_heads, hist_layers, dropout)

        # Command embedding (optional — off when all commands are identical)
        if use_cmd:
            self.cmd_embed = nn.Embedding(3, cmd_embed_dim)
            self.cmd_proj  = _proj(cmd_embed_dim, d)
        ctx_in = d * 3 if use_cmd else d * 2

        # Scene-motion fusion (ablatable)
        if scene_fusion == "rope_xattn":
            self.scene_motion = RoPECrossAttention(
                d, num_heads, GRID_H, GRID_W, dropout, rope_base
            )

        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in, d), nn.LayerNorm(d), nn.GELU()
        )

        fused_dim = d * 3 if use_cmd else d * 2

        # Coarse & fine heads
        self.coarse_heads = nn.ModuleList([
            CoarseHead(fused_dim, d, n_coarse, dropout)
        ] * max(K, 1)) if n_coarse > 0 else None

        self.heads = nn.ModuleList([
            FineHead(d, num_heads, dec_layers, dropout, num_waypoints, causal=causal_decoder)
            for _ in range(max(K, 1))
        ])

        if K > 1:
            router_in = d + K * n_coarse * 2 if n_coarse > 0 else d
            self.router = nn.Sequential(
                nn.Linear(router_in, d), nn.LayerNorm(d), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(),
                nn.Linear(d // 2, K),
            )

        self._init_weights()

    def _init_weights(self):
        own = [self.img_proj, self.patch_proj, self.ctx_proj]
        if self.use_cmd:
            own.append(self.cmd_proj)
        if self.scene_fusion == "rope_xattn":
            own.append(self.scene_motion)
        if self.K > 1:
            own.append(self.router)
        if self.coarse_heads:
            own.extend(self.coarse_heads)
        for m in (mod for mod in (sub for o in own for sub in o.modules())
                  if isinstance(m := mod, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        for head in self.heads:
            nn.init.xavier_uniform_(head.out_proj.weight, gain=0.1)
            nn.init.zeros_(head.out_proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _backbone_features(self, camera):
        feats = self.backbone.forward_features(camera)
        return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]

    def forward(self, camera, history, command):
        B = camera.size(0)

        if self.unfreeze_blocks == 0:
            with torch.no_grad():
                cls, patches = self._backbone_features(camera)
        else:
            cls, patches = self._backbone_features(camera)

        hist_tokens, hist_cls = self.hist_enc(history)
        img_cls  = self.img_proj(cls)
        patches_ = self.patch_proj(patches)

        # Scene-motion fusion: enrich history tokens with spatial patch context
        if self.scene_fusion == "rope_xattn":
            hist_tokens = self.scene_motion(hist_tokens, patches_)
        # scene_fusion == "none": hist_tokens unchanged (image info only via img_cls)

        if self.use_cmd:
            cmd        = self.cmd_proj(self.cmd_embed(command))
            ctx_input  = torch.cat([img_cls, hist_cls, cmd], dim=-1)
            fused      = ctx_input
        else:
            ctx_input  = torch.cat([img_cls, hist_cls], dim=-1)
            fused      = ctx_input

        ctx       = self.ctx_proj(ctx_input)
        ctx_token = ctx.unsqueeze(1)

        coarse_preds = [h(fused) for h in self.coarse_heads] if self.coarse_heads else [None] * max(self.K, 1)
        fine_preds   = [h(hist_tokens, ctx_token) for h in self.heads]

        if self.K > 1 and self.n_coarse > 0:
            coarse_flat   = torch.cat([c.reshape(B, -1) for c in coarse_preds], dim=-1)
            router_logits = self.router(torch.cat([ctx, coarse_flat], dim=-1))
        elif self.K > 1:
            router_logits = self.router(ctx)
        else:
            router_logits = torch.zeros(B, 1, device=camera.device)

        return fine_preds, coarse_preds, router_logits

    def aux_stage_tokens(self):
        return [self._aux_feats[i] for i in self._aux_block_indices]

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())