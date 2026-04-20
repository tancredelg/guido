"""
DrivingPlanner V2
=================
Key changes from V1
--------------------
1. Transformer history encoder   — self-attention over the 21 history steps
                                   instead of GRU, so any step can directly
                                   compare to any other step.

2. Scene encoder context         — backbone CLS token + all 256 patch tokens
                                   are kept and passed downstream.  The
                                   no_grad bug is fixed: frozen/unfrozen is
                                   controlled properly.

3. Scene-motion cross-attention  — the history token sequence attends over
   (fusion)                        patch tokens so spatial road geometry
                                   modulates the history representation
                                   before decoding.  This replaces the old
                                   "motion query attends over patches once"
                                   approach with a richer bidirectional flow.

4. Mixture-of-K decoder          — K=3 transformer decoder heads predict K
                                   candidate trajectories in parallel.  During
                                   training only the best-matching head (min
                                   ADE vs GT) receives gradient — winner-takes-
                                   all.  At inference the head with the lowest
                                   internal confidence score is discarded and
                                   the best is returned (or you can take all K
                                   for test-time ensembling).

5. Smooth trajectory loss        — weighted Huber on the chosen head, plus a
                                   smoothness regulariser on the predicted
                                   trajectory (penalises second-order
                                   acceleration).

Information flow
----------------
camera (B,3,256,256)
    → DINOv3 ViT-B                        cls (B,D), patches (B,256,D)

history (B,21,4)
    → linear embed + pos enc
    → Transformer encoder (self-attn)     hist_tokens (B,21,d)
    → mean-pool                           hist_cls (B,d)

command (B,)
    → nn.Embedding                        cmd (B,d)

Fusion:
    img_proj(cls) + hist_proj(hist_cls) + cmd_proj(cmd)  → ctx (B,d)
    hist_tokens attends over patch_proj(patches)          → hist_tokens' (B,21,d)

Decoder (×K heads, shared backbone):
    60 learned queries + sinusoidal PE
    cross-attend over [hist_tokens' (B,21,d) | ctx_token (B,1,d)]
    → Linear → (B,60,2) for each head k

Training:
    pick head k* = argmin_k ADE(pred_k, gt)
    loss = weighted_huber(pred_k*, gt) + λ * smoothness(pred_k*)

Inference:
    return pred_k*  (k* chosen by intra-head confidence or argmin of own ADE)
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Small building blocks
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
    return pe  # (length, d)


# ─────────────────────────────────────────────────────────────────────────────
# History Transformer encoder
# ─────────────────────────────────────────────────────────────────────────────


class HistoryEncoder(nn.Module):
    """
    Encodes the 21-step history with self-attention so every step can
    directly reference every other step (vs. GRU's left-to-right bottleneck).

    Returns:
        tokens : (B, 21, d)   — full sequence for decoder context
        cls    : (B, d)       — mean-pooled global summary
    """

    def __init__(
        self, in_dim: int = 4, d: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        pe = _sinusoidal_pe(21, d)
        self.register_buffer("pe", pe)  # (21, d)

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

    def forward(self, history: torch.Tensor):
        # history: (B, 21, 4)
        x = self.embed(history) + self.pe.unsqueeze(0)  # (B, 21, d)
        x = self.encoder(x)  # (B, 21, d)
        x = self.norm(x)
        return x, x.mean(dim=1)  # tokens, cls


# ─────────────────────────────────────────────────────────────────────────────
# Scene-motion cross-attention fusion
# ─────────────────────────────────────────────────────────────────────────────


class SceneMotionFusion(nn.Module):
    """
    Let history tokens attend over image patch tokens.

    Query  = history tokens  (B, 21, d)   — what the car has done
    Key/V  = patch tokens    (B, 256, d)  — where the road goes

    Output: enriched history tokens (B, 21, d) where each history step
            has been conditioned on the relevant road patches.

    This is different from the old CrossAttnFusion which collapsed everything
    into a single vector.  Here we keep the full 21-token sequence so the
    decoder can still attend over all 21 steps individually.
    """

    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, hist_tokens: torch.Tensor, patch_tokens: torch.Tensor):
        # hist_tokens:  (B, 21, d)
        # patch_tokens: (B, 256, d)
        q = self.norm_q(hist_tokens)
        kv = self.norm_kv(patch_tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(hist_tokens + out)  # residual, (B, 21, d)


# ─────────────────────────────────────────────────────────────────────────────
# Single trajectory decoder head
# ─────────────────────────────────────────────────────────────────────────────


class TrajectoryHead(nn.Module):
    """
    One trajectory prediction head.  60 learned queries cross-attend over the
    enriched history tokens + a context token from the global representation.

    Context fed to this head: [hist_tokens (B,21,d) | ctx_token (B,1,d)]
    = 22 tokens total.
    """

    def __init__(
        self, d: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1, num_waypoints: int = 60
    ):
        super().__init__()
        self.T = num_waypoints

        self.query_embed = nn.Embedding(num_waypoints, d)
        pe = _sinusoidal_pe(num_waypoints, d)
        self.register_buffer("pe", pe)

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

    def forward(self, hist_tokens: torch.Tensor, ctx_token: torch.Tensor):
        # hist_tokens: (B, 21, d)
        # ctx_token:   (B, 1, d)
        B = hist_tokens.size(0)
        context = torch.cat([hist_tokens, ctx_token], dim=1)  # (B, 22, d)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) + self.pe.unsqueeze(
            0
        )  # (B, 60, d)

        out = self.decoder(queries, context)  # (B, 60, d)
        return self.out_proj(out)  # (B, 60, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main V2 model
# ─────────────────────────────────────────────────────────────────────────────


class DrivingPlannerV2(nn.Module):
    """
    Config keys (all passed from yaml via train_v2.py)
    --------------------------------------------------
    dino_model        : hub name, e.g. 'dinov3_vitb16'
    dino_repo_dir     : path to local dinov3 repo clone
    dino_weights      : path to .pth weights file
    unfreeze_blocks   : int, unfreeze last N ViT-B blocks (default 1)
    d                 : inner model dimension (default 256)
    num_heads         : attention heads (default 4)
    hist_layers       : transformer encoder layers for history (default 2)
    dec_layers        : transformer decoder layers per head (default 2)
    K                 : number of mixture trajectory heads (default 3)
    dropout           : dropout everywhere (default 0.05)
    smoothness_weight : weight for acceleration smoothness regulariser (default 0.1)
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
        dropout: float = 0.05,
        smoothness_weight: float = 0.1,
        num_waypoints: int = 60,
        cmd_embed_dim: int = 64,
    ):
        super().__init__()
        self.K = K
        self.num_waypoints = num_waypoints
        self.smoothness_weight = smoothness_weight
        self.unfreeze_blocks = unfreeze_blocks

        # ── 1. Vision backbone ─────────────────────────────────────────────
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
            n_uf = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            log.info("Unfreezing %d backbone blocks (%d params)", unfreeze_blocks, n_uf)

        dino_dim: int = self.backbone.embed_dim  # 768 for vitb16

        # ── 2. History Transformer encoder ────────────────────────────────
        self.hist_enc = HistoryEncoder(
            in_dim=4,
            d=d,
            num_heads=num_heads,
            num_layers=hist_layers,
            dropout=dropout,
        )

        # ── 3. Command embedding ───────────────────────────────────────────
        self.cmd_embed = nn.Embedding(3, cmd_embed_dim)

        # ── 4. Projection heads → shared dim d ────────────────────────────
        self.img_proj = _proj(dino_dim, d)
        self.patch_proj = nn.Linear(dino_dim, d)  # for cross-attn, no activation
        self.cmd_proj = _proj(cmd_embed_dim, d)

        # ── 5. Scene-motion cross-attention ───────────────────────────────
        self.scene_motion = SceneMotionFusion(d, num_heads, dropout)

        # ── 6. Context projection ──────────────────────────────────────────
        # fuse img_cls + hist_cls + cmd → single context token
        self.ctx_proj = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # ── 7. K trajectory heads ─────────────────────────────────────────
        self.heads = nn.ModuleList(
            [TrajectoryHead(d, num_heads, dec_layers, dropout, num_waypoints) for _ in range(K)]
        )

        # ── 8. Confidence router ──────────────────────────────────────────
        # Robust MLP that predicts which head will produce the best trajectory.
        # Trained with cross-entropy against the WTA winner label from losses.
        # At inference, argmax(router_logits) replaces the oracle.
        self.router = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, K),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────

    def _init_weights(self):
        """
        Only initialise the modules that DrivingPlannerV2 adds itself.
        PyTorch's TransformerEncoderLayer / TransformerDecoderLayer come with
        well-calibrated default inits (Xavier uniform for attention projections,
        Kaiming for FFN).  Overriding them with kaiming(nonlinearity='relu')
        applies gain=√2 to attention Q/K/V projections, making attention logits
        too large → saturated softmax → near-zero gradient.  This is harmless
        for small d but kills training for d≥384 with 3+ layers.
        """
        # Modules whose Linears we explicitly own and want to initialise
        own_modules = [
            self.img_proj,
            self.patch_proj,
            self.cmd_proj,
            self.ctx_proj,
            self.scene_motion,
            self.router,
        ]
        for module in own_modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Output projections (small init keeps predictions near zero at start)
        for head in self.heads:
            nn.init.xavier_uniform_(head.out_proj.weight, gain=0.1)
            nn.init.zeros_(head.out_proj.bias)

        # Embeddings
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Backbone feature extraction ───────────────────────────────────────

    def _backbone_features(self, camera: torch.Tensor):
        feats = self.backbone.forward_features(camera)
        cls = feats["x_norm_clstoken"]  # (B, dino_dim)
        patches = feats["x_norm_patchtokens"]  # (B, 256, dino_dim)
        return cls, patches

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, camera, history, command):
        """
        Returns a list of K tensors, each (B, 60, 2).
        Training code picks the winner; inference takes index 0 of the sorted list.
        """
        B = camera.size(0)

        # 1. Image features
        backbone_frozen = self.unfreeze_blocks == 0
        if backbone_frozen:
            with torch.no_grad():
                cls, patches = self._backbone_features(camera)
        else:
            cls, patches = self._backbone_features(camera)

        # 2. History encoding
        hist_tokens, hist_cls = self.hist_enc(history)  # (B,21,d), (B,d)

        # 3. Command
        cmd = self.cmd_proj(self.cmd_embed(command))  # (B, d)

        # 4. Project image features
        img_cls = self.img_proj(cls)  # (B, d)
        patch_proj = self.patch_proj(patches)  # (B, 256, d)

        # 5. Scene-motion fusion: history tokens attend over patch tokens
        hist_tokens = self.scene_motion(hist_tokens, patch_proj)  # (B, 21, d)

        # 6. Build global context token: img + hist_cls + cmd
        ctx = self.ctx_proj(torch.cat([img_cls, hist_cls, cmd], dim=-1))  # (B, d)
        ctx_token = ctx.unsqueeze(1)  # (B, 1, d)

        # 7. K heads decode in parallel
        preds = [head(hist_tokens, ctx_token) for head in self.heads]

        # 8. Router logits — skip when K=1 (return zero logits, unused)
        if self.K == 1:
            router_logits = torch.zeros(preds[0].size(0), 1, device=preds[0].device)
        else:
            router_logits = self.router(torch.cat([img_cls, hist_cls, cmd], dim=-1).detach())  # (B, K)

        return preds, router_logits

    # ── Utility ───────────────────────────────────────────────────────────

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def backbone_parameters(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def head_parameters(self):
        bb_ids = {id(p) for p in self.backbone.parameters()}
        return (p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids)
