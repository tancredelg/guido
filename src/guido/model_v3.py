"""
DrivingPlannerV3 — velocity prediction + 2D RoPE patch attention.

Key changes vs V2
-----------------
1. Velocity output: fine head predicts per-step velocity (dx, dy);
   trajectory = cumsum(velocity). Forces structural smoothness, makes Huber
   loss work in its quadratic regime everywhere (small dynamic range).
   The coarse head still predicts absolute anchor positions for clarity.

2. 2D RoPE: patch tokens get rotary positional embeddings before entering
   cross-attention with history tokens. Each patch's spatial position is
   encoded by rotating Q/K vectors during attention. Same scheme DINOv3
   uses internally — gives cross-attention real spatial awareness rather
   than treating patches as a bag of vectors.

3. F.scaled_dot_product_attention used directly in the RoPE cross-attention
   path (Flash Attention on modern hardware, no extra library). Other
   transformer modules keep PyTorch's TransformerEncoder/Decoder for
   simplicity — they already use SDPA under the hood when available.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Utilities ─────────────────────────────────────────────────────────────────

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

def _rope_freqs(d_half: int, base: float = 10000.0) -> torch.Tensor:
    """Frequency table for one axis of RoPE. Shape: (d_half,)."""
    return 1.0 / (base ** (torch.arange(0, d_half, dtype=torch.float32) / d_half))


def _build_2d_rope_cache(grid_h: int, grid_w: int, d_head: int, base: float = 10000.0):
    """
    Build cos/sin caches for 2D RoPE on a grid.
    Half the head dim encodes the H axis, the other half the W axis.
    Returns: cos, sin each of shape (grid_h * grid_w, d_head).
    """
    assert d_head % 4 == 0, f"d_head must be divisible by 4 for 2D RoPE, got {d_head}"
    d_half = d_head // 2          # half goes to H axis, half to W axis
    d_quad = d_half // 2          # within each axis, frequencies cover d_half/2 pairs

    freqs = _rope_freqs(d_quad, base)                            # (d_quad,)
    h_pos = torch.arange(grid_h, dtype=torch.float32)             # (H,)
    w_pos = torch.arange(grid_w, dtype=torch.float32)             # (W,)
    h_angles = torch.einsum("i,j->ij", h_pos, freqs)              # (H, d_quad)
    w_angles = torch.einsum("i,j->ij", w_pos, freqs)              # (W, d_quad)

    # Broadcast to (H, W, d_quad) for both axes, interleave to (H, W, d_half)
    H, W = grid_h, grid_w
    h_ang = h_angles.unsqueeze(1).expand(H, W, d_quad)            # (H, W, d_quad)
    w_ang = w_angles.unsqueeze(0).expand(H, W, d_quad)            # (H, W, d_quad)

    # Each axis contributes d_half = 2 * d_quad: pairs (cos, sin) per frequency
    # Final cache layout: [h_freq_0..d_quad-1 | w_freq_0..d_quad-1] each duplicated for cos/sin
    angles = torch.cat([h_ang, w_ang], dim=-1)                    # (H, W, d_half)
    cos = angles.cos().repeat_interleave(2, dim=-1)               # (H, W, d_head)
    sin = angles.sin().repeat_interleave(2, dim=-1)               # (H, W, d_head)
    return cos.reshape(H * W, d_head), sin.reshape(H * W, d_head)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x0,x1,x2,x3,...] → [-x1,x0,-x3,x2,...]"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dim of x by the given cos/sin.
    x:   (..., N, d_head)
    cos: (N, d_head)
    sin: (N, d_head)
    """
    return x * cos + _rotate_half(x) * sin


class RoPEPatchCrossAttention(nn.Module):
    """
    Multi-head cross-attention where the keys (patch tokens) are rotated by
    2D RoPE based on their (h, w) grid position. Queries (history tokens)
    have no positional encoding applied in this module — they keep whatever
    PE the history encoder gave them.

    Uses F.scaled_dot_product_attention for Flash-Attention-quality compute
    on modern PyTorch.
    """
    def __init__(self, d: int, num_heads: int = 4, grid_size: int = 16,
                 dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d % num_heads == 0
        self.d         = d
        self.num_heads = num_heads
        self.d_head    = d // num_heads
        self.grid_size = grid_size

        self.q_proj    = nn.Linear(d, d)
        self.k_proj    = nn.Linear(d, d)
        self.v_proj    = nn.Linear(d, d)
        self.out_proj  = nn.Linear(d, d)
        self.dropout   = dropout

        self.norm_q   = nn.LayerNorm(d)
        self.norm_kv  = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

        # Pre-compute RoPE cache for the patch grid
        cos, sin = _build_2d_rope_cache(grid_size, grid_size, self.d_head, rope_base)
        self.register_buffer("rope_cos", cos)   # (grid*grid, d_head)
        self.register_buffer("rope_sin", sin)

    def forward(self, query: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        query   : (B, Nq, d)  — history tokens
        patches : (B, Nk, d)  — patch tokens, Nk = grid_size**2
        Returns : (B, Nq, d)  — query + cross-attn output (residual + norm)
        """
        B, Nq, _ = query.shape
        _, Nk, _ = patches.shape

        q_in = self.norm_q(query)
        kv_in = self.norm_kv(patches)

        # Project and split into heads: (B, H, N, d_head)
        def split(x):
            return x.reshape(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        q = split(self.q_proj(q_in))
        k = split(self.k_proj(kv_in))
        v = split(self.v_proj(kv_in))

        # Apply RoPE to keys only (patch positional info)
        # rope cos/sin are (Nk, d_head); broadcast over (B, H) leading dims
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # SDPA: uses Flash Attention when available
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )                                                 # (B, H, Nq, d_head)

        # Re-merge heads
        out = attn_out.transpose(1, 2).reshape(B, Nq, self.d)
        out = self.out_proj(out)
        return self.norm_out(query + out)


# ── History encoder (unchanged from V2) ───────────────────────────────────────

class HistoryEncoder(nn.Module):
    def __init__(self, in_dim=4, d=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        self.register_buffer("pe", _sinusoidal_pe(21, d))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, history):
        x = self.embed(history) + self.pe.unsqueeze(0)
        x = self.norm(self.encoder(x))
        return x, x.mean(dim=1)


# ── Coarse head (separate MLP, predicts absolute anchor positions) ────────────

class CoarseHead(nn.Module):
    def __init__(self, fused_dim, d, n_coarse, dropout=0.05):
        super().__init__()
        self.n_coarse = n_coarse
        self.net = nn.Sequential(
            nn.Linear(fused_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, n_coarse * 2),
        )

    def forward(self, fused):
        return self.net(fused).reshape(fused.size(0), self.n_coarse, 2)


# ── Fine head (predicts velocities, integrates to positions) ──────────────────

class FineHead(nn.Module):
    """
    Standard position-predicting fine head. Reverted from velocity — velocity
    prediction showed no meaningful gain and cumsum accumulation hurt static scenarios.
    """
    def __init__(self, d, num_heads=4, dec_layers=3, dropout=0.1, num_waypoints=60):
        super().__init__()
        self.T = num_waypoints
        self.query_embed = nn.Embedding(num_waypoints, d)
        self.register_buffer("pe", _sinusoidal_pe(num_waypoints, d))
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder   = nn.TransformerDecoder(layer, num_layers=dec_layers)
        self.fine_proj = nn.Linear(d, 2)

    def forward(self, hist_tokens, ctx_token):
        B       = hist_tokens.size(0)
        context = torch.cat([hist_tokens, ctx_token], dim=1)
        queries = (self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
                   + self.pe.unsqueeze(0))
        return self.fine_proj(self.decoder(queries, context))


class CausalFineHead(nn.Module):
    """
    Fine head with causal self-attention between decoder queries.

    Standard transformer decoder: each query q_t attends over context
    independently. Query at t=50 has no knowledge of what query t=1 decided.
    This hurts far-horizon coherence — the model can't maintain a consistent
    direction through all 60 steps.

    Fix: after cross-attending over context, apply a causal self-attention
    pass where each query additionally attends to all earlier queries.
    q_t can now ask "given my context reading AND what queries 1..t-1 decided,
    what should I predict?" This enforces temporal coherence without autoregression.

    Implementation: one TransformerDecoder for context cross-attn, then one
    TransformerEncoder with causal mask for query self-attn.
    """
    def __init__(self, d, num_heads=4, dec_layers=2, causal_layers=1, dropout=0.1, num_waypoints=60):
        super().__init__()
        self.T = num_waypoints
        self.query_embed = nn.Embedding(num_waypoints, d)
        self.register_buffer("pe", _sinusoidal_pe(num_waypoints, d))

        # Stage 1: cross-attend over context (history + image + command)
        ctx_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.ctx_decoder = nn.TransformerDecoder(ctx_layer, num_layers=dec_layers)

        # Stage 2: causal self-attention — q_t attends to q_0..q_{t-1}
        causal_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.causal_enc = nn.TransformerEncoder(causal_layer, num_layers=causal_layers)

        self.fine_proj = nn.Linear(d, 2)

        # Causal mask: position i can attend to positions 0..i only
        mask = torch.triu(torch.ones(num_waypoints, num_waypoints), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)   # True = masked out

    def forward(self, hist_tokens, ctx_token):
        B       = hist_tokens.size(0)
        context = torch.cat([hist_tokens, ctx_token], dim=1)
        queries = (self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
                   + self.pe.unsqueeze(0))

        # Cross-attend over scene context
        after_ctx = self.ctx_decoder(queries, context)       # (B, T, d)

        # Causal self-attention for temporal coherence
        after_causal = self.causal_enc(
            after_ctx,
            mask=self.causal_mask,                           # (T, T) bool
        )                                                     # (B, T, d)

        return self.fine_proj(after_causal)                  # (B, T, 2)


# ── Main model ────────────────────────────────────────────────────────────────

class DrivingPlannerV3(nn.Module):
    """
    Same config keys as V2, plus:
      rope_grid_size : spatial grid size for 2D RoPE (default 16, matches
                       DINOv3 ViT-B/16 patch grid for 256×256 input)
      rope_base      : RoPE frequency base (default 10000)
    """

    def __init__(
        self,
        *,
        dino_model: str       = "dinov3_vitb16",
        dino_repo_dir: str    = "",
        dino_weights: str     = "",
        unfreeze_blocks: int  = 1,
        d: int                = 256,
        num_heads: int        = 4,
        hist_layers: int      = 2,
        dec_layers: int       = 2,
        causal_layers: int    = 1,      # causal self-attn layers in decoder (0 = standard)
        K: int                = 1,
        n_coarse: int         = 5,
        dropout: float        = 0.05,
        smoothness_weight: float = 0.05,
        num_waypoints: int    = 60,
        cmd_embed_dim: int    = 128,
        rope_grid_size: int   = 16,
        rope_base: float      = 10000.0,
        speed_cond: bool      = True,   # condition decoder on observed speed scalar
    ):
        super().__init__()
        self.K                 = K
        self.n_coarse          = n_coarse
        self.num_waypoints     = num_waypoints
        self.smoothness_weight = smoothness_weight
        self.unfreeze_blocks   = unfreeze_blocks
        self.speed_cond        = speed_cond
        self.causal_layers     = causal_layers

        # Backbone
        if not dino_repo_dir or not dino_weights:
            raise ValueError("dino_repo_dir and dino_weights must be set.")
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

        # Encoders
        self.hist_enc   = HistoryEncoder(4, d, num_heads, hist_layers, dropout)
        self.cmd_embed  = nn.Embedding(3, cmd_embed_dim)
        self.img_proj   = _proj(dino_dim, d)
        self.patch_proj = nn.Linear(dino_dim, d)
        self.cmd_proj   = _proj(cmd_embed_dim, d)

        # Scene-motion fusion: now uses 2D RoPE on patches
        self.scene_motion = RoPEPatchCrossAttention(
            d, num_heads, rope_grid_size, dropout, rope_base,
        )

        self.ctx_proj = nn.Sequential(
            nn.Linear(d * 3, d), nn.LayerNorm(d), nn.GELU(),
        )
        # Speed conditioning: observed speed from last 2 history steps → scalar
        # projected to d and added to ctx before fine decoder.
        # Lets the model learn "speed≈0 → predict near-zero displacement."
        if speed_cond:
            self.speed_proj = nn.Sequential(
                nn.Linear(1, d // 4), nn.GELU(), nn.Linear(d // 4, d),
            )
        fused_dim = d * 3 + (d if speed_cond else 0)

        # Coarse heads
        self.coarse_heads = nn.ModuleList([
            CoarseHead(fused_dim, d, n_coarse, dropout)
            for _ in range(max(K, 1))
        ]) if n_coarse > 0 else None

        # Fine heads — causal if causal_layers > 0, else standard
        if causal_layers > 0:
            self.heads = nn.ModuleList([
                CausalFineHead(d, num_heads, dec_layers, causal_layers,
                               dropout, num_waypoints)
                for _ in range(max(K, 1))
            ])
        else:
            self.heads = nn.ModuleList([
                FineHead(d, num_heads, dec_layers, dropout, num_waypoints)
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
        own = [self.img_proj, self.patch_proj, self.cmd_proj,
               self.ctx_proj, self.scene_motion]
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
            out_proj = getattr(head, "fine_proj", getattr(head, "vel_proj", None))
            if out_proj is not None:
                nn.init.xavier_uniform_(out_proj.weight, gain=0.1)
                nn.init.zeros_(out_proj.bias)
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
        cmd     = self.cmd_proj(self.cmd_embed(command))
        img_cls = self.img_proj(cls)
        # RoPE applied internally in scene_motion via key rotation
        hist_tokens = self.scene_motion(hist_tokens, self.patch_proj(patches))

        ctx       = self.ctx_proj(torch.cat([img_cls, hist_cls, cmd], dim=-1))
        ctx_token = ctx.unsqueeze(1)

        # Speed conditioning: L2 speed from last 2 history steps (x,y cols)
        if self.speed_cond:
            # history: (B, 21, 4) — [x, y, sin_h, cos_h]
            speed = torch.norm(
                history[:, -1, :2] - history[:, -2, :2], p=2, dim=-1, keepdim=True
            )                                                       # (B, 1)
            speed_emb = self.speed_proj(speed)                      # (B, d)
            fused = torch.cat([img_cls, hist_cls, cmd, speed_emb], dim=-1)
            # Also add to ctx_token so decoder sees speed
            ctx_token = (ctx + speed_emb).unsqueeze(1)
        else:
            fused = torch.cat([img_cls, hist_cls, cmd], dim=-1)

        if self.coarse_heads is not None:
            coarse_preds = [h(fused) for h in self.coarse_heads]
        else:
            coarse_preds = [None] * max(self.K, 1)

        fine_preds = [h(hist_tokens, ctx_token) for h in self.heads]

        if self.K > 1 and self.n_coarse > 0:
            coarse_flat   = torch.cat([c.reshape(B, -1) for c in coarse_preds], dim=-1)
            router_logits = self.router(torch.cat([ctx, coarse_flat], dim=-1))
        elif self.K > 1:
            router_logits = self.router(ctx)
        else:
            router_logits = torch.zeros(B, 1, device=camera.device)

        return fine_preds, coarse_preds, router_logits

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())