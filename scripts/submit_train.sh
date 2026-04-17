#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_train.sh
# Monitor with: squeue -u $USER
#               tail -f logs/<job_id>.out

#SBATCH --job-name=guido_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                  # matches num_workers in config
#SBATCH --mem=32G
#SBATCH --time=04:00:00                    # 60 epochs typically finishes in ~2h
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

SCRATCH=/scratch/izar/$USER

# ── Environment ───────────────────────────────────────────────────────────────
# # Activate the uv-managed virtual environment
# source .venv/bin/activate

# Keep HuggingFace weights on scratch (fast I/O, no $HOME quota issues).
# Pre-cache on a login node first (needs internet):
#   HF_HOME=$SCRATCH/.cache/huggingface python -c \
#     "from transformers import AutoModel; \
#      AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')"
# export HF_HOME=$SCRATCH/.cache/huggingface

# ── DINOv3 paths ──────────────────────────────────────────────────────────────
# Set these to wherever you cloned the repo and downloaded the weights.
# Only needs to be done once per cluster account:
#   git clone https://github.com/facebookresearch/dinov3.git $SCRATCH/dinov3
#   wget -O $SCRATCH/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth \
#        '<URL from Meta access email>'
DINO_REPO=$SCRATCH/dinov3
DINO_WEIGHTS=$SCRATCH/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth
 

# ── Data ──────────────────────────────────────────────────────────────────────
# Data lives on scratch for fast I/O; override the config path via CLI flag.
DATA_DIR=$SCRATCH/CIVIL-459/guido/data

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data:"
echo "$(du -sh $DATA_DIR)"

uv run src/train.py \
    --config  configs/baseline.yaml \
    --data-dir "$DATA_DIR"

echo "Job finished at $(date)"