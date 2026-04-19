#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_eval.sh
# Monitor with: squeue -u $USER
#               tail -f logs/eval_<job_id>.out

#SBATCH --job-name=guido_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                  # matches num_workers in config
#SBATCH --mem=32G
#SBATCH --time=04:00:00                    # 60 epochs typically finishes in ~2h
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
mkdir -p logs

SCRATCH=/scratch/izar/$USER


echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

uv run src/predict.py \
    --checkpoint checkpoints/run_20260417_2107_epoch058_ade1.9614.pth \
    --split val \
    --visualize

echo "Job finished at $(date)"
