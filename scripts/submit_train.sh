#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_train.sh
# Monitor with: squeue -u $USER
#               tail -f logs/<job_id>.out

#SBATCH --job-name=guido_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6                  # config's num_workers + 2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/V2/%j.out
#SBATCH --error=logs/V2/%j.err

set -euo pipefail
mkdir -p logs

SCRATCH=/scratch/izar/$USER

# ── DINOv3 paths ──────────────────────────────────────────────────────────────
# Set these to wherever you cloned the repo and downloaded the weights.
# Only needs to be done once per cluster account:
#   git clone https://github.com/facebookresearch/dinov3.git $SCRATCH/dinov3
#   wget -O $SCRATCH/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth \
#        '<URL from Meta access email>'
# DINO_REPO=$SCRATCH/dinov3
# DINO_WEIGHTS=$SCRATCH/dinov3/weights/dinov3_vits16_pretrain_lvd1689m.pth
 

# ── Data ──────────────────────────────────────────────────────────────────────
# Data lives on scratch for fast I/O; override the config path via CLI flag.
DATA_DIR=$SCRATCH/CIVIL-459/guido/data

CFG=configs/V2/baseline.yaml

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

uv run src/train_v2.py \
    --config   "$CFG" \
    --data-dir "$DATA_DIR"

echo "Job finished at $(date)"