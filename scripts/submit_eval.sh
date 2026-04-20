#!/bin/bash
# ── SLURM job script for EPFL SCITAS ─────────────────────────────────────────
# Submit with:  sbatch scripts/submit_eval.sh
# Monitor with: squeue -u $USER
#               tail -f logs/eval_<job_id>.out

#SBATCH --job-name=guido_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/V2/eval_%j.out
#SBATCH --error=logs/V2/eval_%j.err

set -euo pipefail
mkdir -p logs

SCRATCH=/scratch/izar/$USER

DATA_DIR=$SCRATCH/CIVIL-459/guido/data

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data:"
echo "$(du -sh $DATA_DIR)"

uv run src/predict_v2.py \
    --checkpoint 'checkpoints/run_20260419_2222_epoch150_ade1.5985.pth' \
    --split val \
    --data-dir "$DATA_DIR" \
    --visualize \
    --vis-output predictions_v2.pdf

echo "Job finished at $(date)"
