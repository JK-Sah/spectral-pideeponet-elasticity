#!/bin/bash -l
#SBATCH --job-name=r1C-hetx
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-35
#SBATCH --output=%x-%A_%a.out
# Grid extension for the smooth heterogeneous spectral operator at M = 32/64/100:
# with {1e-3, 1e-2, 1e-1} the optimum fell at an end of the grid (0.1 for
# M = 32 and 64, 1e-3 for M = 100).  Adds {0, 1e-4, 1, 10}, the span used for M = 16.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44); S=${SEEDS[$((T % 3))]}; I=$((T / 3))
MS=(32 64 100); CS=(32 48 64); WS=(0 1e-4 1 10)
M=${MS[$((I / 4))]}; CM=${CS[$((I / 4))]}; W=${WS[$((I % 4))]}
python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --epochs 4000 --batch 32 \
  --eval_every 20 --depth 4 --device cpu --hidden 192 --modes $M --cos_modes $CM \
  --seed $S --w_pde $W --out $O/routeb/smooth
