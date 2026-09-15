#!/bin/bash -l
#SBATCH --job-name=cmame-timingv2
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# 8-thread configuration (matches the ledger hardware)
python timing_v2.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
  --ckpt_dir "$PROJ/runs/nonlinear_v2" --threads 8 --seed 111 \
  --out "$PROJ/runs/nonlinear/timing_v2_t8.json"
# single-thread configuration (clean apples-to-apples single-query latency)
python timing_v2.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
  --ckpt_dir "$PROJ/runs/nonlinear_v2" --threads 1 --seed 111 \
  --out "$PROJ/runs/nonlinear/timing_v2_t1.json"
echo TIMINGV2_DONE
