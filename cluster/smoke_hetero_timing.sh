#!/bin/bash -l
#SBATCH --job-name=cmame-smkhet
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "### smoke: smooth field, 2 ranks, 2 reps -- correctness only, NOT for the paper"
python timing_hetero_all.py --data "$PROJ/data/hetero_field.npz" \
  --ckpt_dir "$PROJ/runs/route_b" --ranks 16 64 --warm 1 --reps 2 \
  --out /tmp/smoke_smooth_$$.json
echo "### smoke: rough field, 2 ranks, 2 reps"
python timing_hetero_all.py --data "$PROJ/data_rough/hetero_field.npz" \
  --ckpt_dir "$PROJ/runs/rough" --ranks 16 512 --warm 1 --reps 2 \
  --out /tmp/smoke_rough_$$.json
echo SMOKE_HETERO_OK
