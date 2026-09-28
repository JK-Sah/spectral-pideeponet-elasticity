#!/bin/bash -l
#SBATCH --job-name=cmame-tall
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "### node"; hostname
lscpu | grep -E "Model name|Socket|Thread|CPU max" || true

echo; echo "### classical baselines, both benchmarks, one protocol"
python timing_classical_all.py --threads 4 \
  --het_data "$PROJ/data/hetero_field.npz" \
  --out "$PROJ/runs/linear/timing_classical_all.json"

for M in 16 64; do
  echo; echo "### learned models, M=$M, all protocol conditions"
  python timing_linear_v3.py --ckpt_dir "$PROJ/runs/linear/ckpt" --trunk $M \
    --threads 4 --order both \
    --out "$PROJ/runs/linear/timing_linear_v3_M${M}.json"
done
echo TIMINGALL_DONE
