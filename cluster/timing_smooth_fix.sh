#!/bin/bash -l
#SBATCH --job-name=cmame-tsm
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
hostname
# All four smooth-field checkpoints, keyed individually: the previous run kept
# only whichever the glob returned last, which was the data-only variant.
python timing_hetero_all.py --data "$PROJ/data/hetero_field.npz" \
  --ckpt_dir "$PROJ/runs/route_b" --ranks 8 16 32 64 128 256 \
  --out "$PROJ/runs/hetero/timing_smooth_v2.json"
echo TIMINGSMOOTHFIX_DONE
