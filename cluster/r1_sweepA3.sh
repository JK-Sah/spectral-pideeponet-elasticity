#!/bin/bash -l
#SBATCH --job-name=r1-swA3
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=%x-%A_%a.out
# Upper edge of the heterogeneous residual-weight sweep: stage A put the
# validation optimum at the largest weight tried (1e-1).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; W=(0.3 1 3 10); S=(42 43 44)
python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --modes 16 --cos_modes 16 \
  --hidden 192 --depth 4 --epochs 800 --batch 32 --eval_every 20 --device cpu \
  --w_pde ${W[$((T / 3))]} --seed ${S[$((T % 3))]} --out $PROJ/runs/r1/hetero_wsweep
