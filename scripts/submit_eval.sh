#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_eval.sh
# Monitor with: squeue -u $USER
#               tail -f logs/eval_<job_id>.out

#SBATCH --job-name=guido_eval
#SBATCH --partition=mig12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=11800M
#SBATCH --time=01:00:00
#SBATCH --output=logs/V4/eval_%j.out
#SBATCH --error=logs/V4/eval_%j.err

set -euo pipefail
mkdir -p logs/V4/

SCRATCH=/scratch/$USER/tanc

DATA_DIR=$SCRATCH/guido/data

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data:"
echo "$(du -sh $DATA_DIR)"

uv run src/predict_v4.py \
    --checkpoint 'checkpoints/run_20260513_2113_epoch189_ade1.4892.pth' \
    --split test \
    --data-dir "$DATA_DIR" \
    --visualize \
    --vis-output predictions_v4.pdf

echo "Job finished at $(date)"
