# Guido — End-to-End Trajectory Prediction for Autonomous Driving

A crowd-favourite Cars character, now re-incarnated as an end-to-end neural 
trajectory planner that predicts 60 future waypoints from a single front-facing 
camera image, ego-motion history, and a high-level driving command. Trained and 
evaluated on a subset of the [nuPlan](https://www.nuplan.org/) dataset.

**Phase 1 result:**, val ADE ≈ 1.53 m over a 60-step horizon.

---

## Architecture

```
Camera (3×256×256)  →  DINOv3 ViT-B/16  →  CLS token + 256 patch tokens
                        (last block unfrozen)

History (21 steps)  →  Transformer encoder  →  history tokens + summary
Command             →  Embedding

Scene-motion grounding:
  history tokens attend over patch tokens  →  2D RoPE cross-attention
  (each history step queries the spatial image grid with positional awareness)

Context token:  linear proj of [img_cls | hist_cls | cmd]

Coarse head (MLP):  5 anchor waypoints  +  auxiliary MSE loss
Fine head (Transformer decoder, 2L):
  60 learned queries cross-attend over [grounded history tokens | context token]
  → 60-step trajectory (B, 60, 2)
```

The scene-motion cross-attention uses 2D RoPE on the patch token keys,
matching the positional encoding scheme used internally by the DINOv3 backbone.
This replaced a simpler V1 cross-attention variant (single motion query over
patches) with a richer design where the entire 21-step history sequence is
spatially grounded before decoding. The coarse MLP head provides auxiliary
gradient at intermediate timesteps without reducing fine-decoder layer capacity.

---

## Results

| Model | Val ADE |
|---|---|
| V1 baseline (ViT-S, GRU, MLP decoder) | 1.96 |
| V1 + ViT-B backbone | 1.90 |
| V2 (Transformer history encoder) | 1.57 |
| V2 large + coarse-to-fine | 1.54 |
| **V3 (2D RoPE, best Phase 1)** | **1.53** |

---

## Installation

```bash
git clone https://github.com/tancredelg/guido
cd guido
uv sync
```

**DINOv3 weights**: request access from Meta via the
[DINOv3 GitHub repo](https://github.com/facebookresearch/dinov3).
Download `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` and point
`dino_repo_dir` / `dino_weights` in your config to its location.

**Dataset**: a 2.6 GB subset of nuPlan (5k train / 1k val / 1k test samples).
For reproduction with the exact splits used here, see the
[Kaggle competition](https://www.kaggle.com/competitions/dlav-2026-phase-1/leaderboard).
For broader experiments, the full [nuPlan dataset](https://www.nuplan.org/)
can be used with minor changes to `dataset.py`.

**Hardware**: training runs on a single V100 (32 GB) in 2–4 hours.
Logging uses [Weights & Biases](https://wandb.ai) — set `wandb_entity` in
your config or pass `--no-wandb` to disable.

---

## Usage

```bash
# Train
uv run src/train_v3.py --config configs/V3/baseline.yaml

# Validate (reports ADE/FDE locally — honest estimate of Kaggle score)
uv run src/predict_v3.py --checkpoint checkpoints/best.pth --split val

# Generate submission CSV
uv run src/predict_v3.py --checkpoint checkpoints/best.pth --split test \
    --output submission.csv

# Test-time augmentation (mirror flip + average, potentially free tiny ADE gain)
uv run src/predict_v3.py --checkpoint checkpoints/best.pth --split test \
    --tta --output submission_tta.csv

# Visualise predictions
uv run src/predict_v3.py --checkpoint checkpoints/best.pth --split val \
    --visualize --vis-output predictions.pdf
```

For SLURM clusters, see `scripts/submit_train.sh` — edit the `CFG` and
`DATA_DIR` variables and submit with `sbatch scripts/submit_train.sh`.

---

## Experiments

Three rounds of architecture and training changes, each building on the previous.

### V1 — Baseline architecture

Standard encode-fuse-decode pipeline: frozen DINOv3 ViT-S CLS token + GRU
history encoder + concat fusion + MLP decoder. ADE ~1.96.

Ablations: ViT-B backbone (✓ −0.06 ADE), data augmentation (✗ aggressive
augmentation hurt), wider hidden dims (✗ overfit on 5k samples), cross-attention
fusion (✗ marginal data), transposed-conv and transformer decoders (neither
conclusively better than MLP at this scale).

Several bugs were fixed during this phase that had invalidated earlier
experiments:
- **Huber delta regression** — `delta=0.1` accidentally left active for
  absolute-position targets, collapsing gradients for errors > 10 cm.
- **`torch.no_grad()` unfreeze bug** — backbone blocks were wrapped
  unconditionally, silencing gradients through supposedly unfrozen blocks.
- **Kaiming init on transformer attention** — applying gain=√2 to Q/K/V
  projections caused attention saturation at d≥384; fixed by leaving
  PyTorch defaults on attention internals.

### V2 — Transformer history encoder + mixture heads

Replaced GRU with a Transformer encoder so history steps can attend to each
other directly. Added a separate MLP coarse head for auxiliary supervision.
Best single-head result: ADE ~1.54.

Explored K=3..6 mixture heads with winner-takes-all training. Oracle
best-of-K ADE reached **~1.0–1.2 with K=6**, demonstrating real head
specialisation. However, a trained router could only close the
gap to ~1.65–1.7 at inference. The 5k dataset proved insufficient to jointly
train specialised trajectory heads and a quality routing head — the router
CE loss competed with WTA specialisation gradient, and increasing K or
router capacity did not resolve this. The most effective approaches attempted
were trajectory-conditioned routing (router sees coarse predictions from each
head, not just input features) and a two-phase training schedule (freeze
router until heads specialise), both of which helped but left a large gap
to the oracle.

### V3 — 2D RoPE + targeted improvements

Added 2D RoPE to patch cross-attention. Tested velocity prediction (no
meaningful gain over position prediction, cumsum accumulation hurt static
scenarios), causal self-attention between decoder queries (slightly worse),
speed scalar conditioning (marginal), and various endpoint-anchored loss
variants (U-shaped weighting, explicit FDE term — none helped reliably).

Final result: ADE **1.53** (val), 1st place on Phase 1 leaderboard.

---

## Repository layout

```
src/
  train.py / predict.py        V1 training and inference
  train_v2.py / predict_v2.py  V2 training and inference
  train_v3.py / predict_v3.py  V3 training and inference  ← use these
  guido/
    dataset.py    DrivingDataset, augmentation, weighted sampler
    model.py      V1 DrivingPlanner
    model_v2.py   V2 DrivingPlannerV2 (Transformer encoder + mixture heads)
    model_v3.py   V3 DrivingPlannerV3 (+ 2D RoPE, speed conditioning)
    losses.py     V1 losses
    losses_v2.py  V2/V3 losses (WTA, coarse auxiliary, smoothness, FDE)
    utils.py      Checkpointing and submission CSV builder

configs/
  V3/final.yaml      Best Phase 1 config  ← start here
  V1/                V1 ablation suite
  V2/                V2 architecture experiments
  V3/                V3 experiments
```

---

## Reference papers

Papers consulted during this project are in `docs/`:
- TransFuser (Prakash et al., 2021) — transformer-based sensor fusion for AD
- UniAD (Hu et al., 2022) — unified autonomous driving architecture
- VAD (Jiang et al., 2023) — vectorised scene representation for AD