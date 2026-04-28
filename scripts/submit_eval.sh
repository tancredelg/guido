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
#SBATCH --output=logs/V3/eval_%j.out
#SBATCH --error=logs/V3/eval_%j.err

set -euo pipefail
mkdir -p logs/V3/

SCRATCH=/scratch/$USER/tanc

DATA_DIR=$SCRATCH/guido/data

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data:"
echo "$(du -sh $DATA_DIR)"

uv run src/predict_v3.py \
    --checkpoint 'checkpoints/run_20260427_1629_epoch381_ade1.5259.pth' \
    --split test \
    --data-dir "$DATA_DIR" \
    --visualize \
    --vis-output predictions_v3.pdf

echo "Job finished at $(date)"
