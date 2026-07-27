#!/bin/bash
#SBATCH --job-name=lebo_clasp
#SBATCH --partition=common
#SBATCH --qos=kjurasz
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lebo_clasp_%j.out
#SBATCH --error=logs/lebo_clasp_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/pep-compass}"
mkdir -p logs results/lebo_clasp

PYTHON="${HOME}/pep-compass/.venv/bin/python"

echo "=========================================="
echo "LE-BO CLASP (product, lambda=0.1, S. aureus)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started: $(date)"
echo "=========================================="

"${PYTHON}" scripts/runner/run_optimization.py \
  --config configs/optimization/experiments/lebo/clasp.json \
  --device cuda:0

echo "Finished: $(date)"
