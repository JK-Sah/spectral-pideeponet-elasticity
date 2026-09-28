#!/bin/bash -l
#SBATCH --job-name=cmame-tlin3
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
lscpu | grep -E "Model name|Socket|Thread|MHz" || true
for M in 16 64; do
  echo "================ M=$M ================"
  python timing_linear_v3.py --ckpt_dir "$PROJ/runs/linear/ckpt" --trunk $M --threads 4 --order both \
    --out "$PROJ/runs/linear/timing_linear_v3_M${M}.json"
done
echo TIMINGV3_ALL_DONE
