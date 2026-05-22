#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_eval.sh
# Monitor with: squeue -u $USER
#               tail -f logs/eval_<job_id>.out

#SBATCH --job-name=guido_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=11800M
#SBATCH --time=01:00:00
#SBATCH --output=logs/V5/eval_%j.out
#SBATCH --error=logs/V5/eval_%j.err

set -euo pipefail
mkdir -p logs/V5/

SCRATCH=/scratch/$USER/tanc

DATA_DIR=$SCRATCH/guido/data

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data:"
echo "$(du -sh $DATA_DIR)"

uv run src/predict_v5.py \
    --checkpoint 'checkpoints/run_20260521_2210_epoch226_ade1.2616.pth' \
    --split test \
    --data-dir "$DATA_DIR" \
    --visualize \
    --vis-output predictions_v5.pdf

echo "Job finished at $(date)"
