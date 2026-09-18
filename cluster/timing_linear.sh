#!/bin/bash -l
#SBATCH --job-name=cmame-tlin
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python timing_linear.py --ckpt_dir "$PROJ/runs/linear/ckpt" --trunk 16 --threads 4 \
  --out "$PROJ/runs/linear/timing_linear_M16.json"
echo TIMINGLINEAR_DONE
