"""
DrivingPlanner – Milestone 1 (DINOv3 edition)
===============================================
Architecture summary
--------------------
1. Image encoder  : frozen DINOv3-small (ViT-S/16), loaded via torch.hub from a
                    local clone of the dinov3 repo + a local .pth weights file
                    (obtained by requesting access at ai.meta.com/dinov3).
                    Produces a 384-d CLS token representing the current scene.
                    Frozen because we only have 5 k training samples.
2. History encoder: 2-layer GRU over the 21-step (x, y, sin h, cos h) history.
3. Command embed  : nn.Embedding(3, cmd_embed_dim) for forward/left/right.
4. Projection     : each modality → a common `fusion_dim` via Linear+LN+GELU.
5. Fusion         : simple concatenation of the three projected vectors.
6. Decoder        : 3-layer MLP → (60, 2) future waypoints (x, y in ego-frame).

Setup on SCITAS
---------------
1. Clone the dinov3 repo somewhere on scratch:
       cd $SCRATCH && git clone https://github.com/facebookresearch/dinov3.git

2. Download the ViT-S/16 weights using the URL from Meta's access email:
       wget -O $SCRATCH/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth \\
            "<URL_FROM_META_EMAIL>"

3. Set these two paths in configs/baseline.yaml (or override via CLI):
       dino_repo_dir: /scratch/.../dinov3
       dino_weights:  /scratch/.../dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth

The SLURM script exports them as env vars so you only need to edit one place.
"""

import torch
import torch.nn as nn


class DrivingPlanner(nn.Module):
    def __init__(
        self,
        *,
        dino_model: str = "dinov3_vits16",  # hub entry-point name
        dino_repo_dir: str = "",  # local clone of dinov3 repo
        dino_weights: str = "",  # path to downloaded .pth file
        hist_input_dim: int = 4,  # x, y, sin(h), cos(h)
        hist_hidden_dim: int = 128,
        hist_num_layers: int = 2,
        cmd_embed_dim: int = 32,
        fusion_dim: int = 256,
        num_waypoints: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints

        # ── 1. Image backbone ──────────────────────────────────────────────────
        if not dino_repo_dir or not dino_weights:
            raise ValueError(
                "Both dino_repo_dir and dino_weights must be set.\n"
                "  dino_repo_dir : local path to the cloned dinov3 repo\n"
                "  dino_weights  : local path to the downloaded .pth file\n"
                "Set them in configs/baseline.yaml or pass via --config."
            )
        self.backbone = torch.hub.load(
            dino_repo_dir,
            dino_model,
            source="local",
            weights=dino_weights,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        # DINOv3 ViT models expose .embed_dim just like DINOv2.
        # ViT-S → 384, ViT-B → 768, ViT-L → 1024.
        dino_dim: int = self.backbone.embed_dim

        # ── 2. History encoder ─────────────────────────────────────────────────
        self.history_gru = nn.GRU(
            input_size=hist_input_dim,
            hidden_size=hist_hidden_dim,
            num_layers=hist_num_layers,
            batch_first=True,
            dropout=dropout if hist_num_layers > 1 else 0.0,
        )

        # ── 3. Command embedding ───────────────────────────────────────────────
        self.cmd_embed = nn.Embedding(num_embeddings=3, embedding_dim=cmd_embed_dim)

        # ── 4. Projection heads (each modality → fusion_dim) ──────────────────
        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )

        self.img_proj = _proj(dino_dim)
        self.hist_proj = _proj(hist_hidden_dim)
        self.cmd_proj = _proj(cmd_embed_dim)

        # ── 5. Decoder (concat → waypoints) ───────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_waypoints * 2),
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Kaiming-uniform for linear layers; orthogonal for GRU recurrent weights."""
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

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        camera: torch.Tensor,  # (B, 3, 256, 256)  preprocessed, normalised
        history: torch.Tensor,  # (B, 21, 4)         [x, y, sin(h), cos(h)]
        command: torch.Tensor,  # (B,)  long          0=forward, 1=left, 2=right
    ) -> torch.Tensor:  # (B, 60, 2)          predicted (x, y) waypoints

        # 1. Image → CLS token  (B, embed_dim)
        #    torch.hub DINOv3 backbone: calling model(x) returns the CLS token,
        #    identical to the DINOv2 torch.hub behaviour.
        with torch.no_grad():
            img_feat = self.backbone(camera)

        # 2. History → last GRU hidden state  (B, hist_hidden_dim)
        _, h_n = self.history_gru(history)
        hist_feat = h_n[-1]  # top layer's hidden state

        # 3. Command → embedding  (B, cmd_embed_dim)
        cmd_feat = self.cmd_embed(command)

        # 4. Project all to fusion_dim
        img_feat = self.img_proj(img_feat)
        hist_feat = self.hist_proj(hist_feat)
        cmd_feat = self.cmd_proj(cmd_feat)

        # 5. Fuse: concat → (B, fusion_dim * 3)
        fused = torch.cat([img_feat, hist_feat, cmd_feat], dim=-1)

        # 6. Decode → (B, 60, 2)
        out = self.decoder(fused)
        return out.reshape(-1, self.num_waypoints, 2)

    # ── Utility ────────────────────────────────────────────────────────────────

    def trainable_parameters(self):
        """Yields only non-frozen parameters (excludes the DINOv3 backbone)."""
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
