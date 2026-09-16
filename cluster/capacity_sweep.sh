#!/bin/bash -l
#SBATCH --job-name=cmame-cap
#SBATCH --account=flowlab
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
mkdir -p "$PROJ/runs/linear"
for M in 16 32 64; do
  echo "=== trunk capacity M=$M ==="
  python canonical_linear.py --seeds 42 43 44 --trunk $M --device auto \
    --ckpt_dir "$PROJ/runs/linear/ckpt" \
    --out "$PROJ/runs/linear/canonical_M${M}.json"
done
echo CAPACITY_DONE
