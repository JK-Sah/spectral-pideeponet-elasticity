#!/bin/bash -l
#SBATCH --job-name=cmame-metrics
#SBATCH --account=flowlab
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python canonical_metrics.py --seeds 42 43 44 --device auto \
  --out "$PROJ/runs/linear/canonical_metrics.json"
echo METRICS_DONE
