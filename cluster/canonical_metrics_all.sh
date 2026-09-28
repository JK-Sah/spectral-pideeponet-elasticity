#!/bin/bash -l
#SBATCH --job-name=cmame-metall
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for M in 16 64; do
  echo "================ M=$M ================"
  python canonical_metrics_all.py --ckpt_dir "$PROJ/runs/linear/ckpt" --trunk $M \
    --out "$PROJ/runs/linear/canonical_metrics_all_M${M}.json"
done
echo METRICSALL_DONE
