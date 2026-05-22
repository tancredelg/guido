#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_train.sh
# Monitor with: squeue -u $USER
#               tail -f logs/<job_id>.out

#SBATCH --job-name=guido_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/V5/%j.out
#SBATCH --error=logs/V5/%j.err

SCRATCH=/scratch/$USER/tanc

set -euo pipefail
mkdir -p $SCRATCH/guido/logs/V5/

# ── DINOv3 paths ──────────────────────────────────────────────────────────────
# Set these to wherever you cloned the repo and downloaded the weights.
# Only needs to be done once per cluster account:
#   git clone https://github.com/facebookresearch/dinov3.git $SCRATCH/dinov3
#   wget -O $SCRATCH/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth \
#        '<URL from Meta access email>'

# ── Data ──────────────────────────────────────────────────────────────────────
# Data lives on scratch for fast I/O; override the config path via CLI flag.
DATA_DIR=$SCRATCH/guido/data

CFG='configs/V5/phase3.yaml'

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data:"
echo "$(du -sh $DATA_DIR)"
echo "Config: $CFG"

# Nuke stale bytecode before every job to avoid import errors after code changes
# find src/ -name "*.pyc" -delete
# find src/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# find src/guido -name "*.pyc" -delete
# find src/guido -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

uv run src/train_v5.py \
    --config   "$CFG" \
    --data-dir "$DATA_DIR"

echo "Job finished at $(date)"