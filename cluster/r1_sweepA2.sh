#!/bin/bash -l
#SBATCH --job-name=r1-swA2
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
# Lower edge of the heterogeneous residual-weight sweep (w = 1e-5).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=(42 43 44)
python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --modes 16 --cos_modes 16 \
  --hidden 192 --depth 4 --epochs 800 --batch 32 --eval_every 20 --device cpu \
  --w_pde 1e-5 --seed ${S[$SLURM_ARRAY_TASK_ID]} --out $PROJ/runs/r1/hetero_wsweep
