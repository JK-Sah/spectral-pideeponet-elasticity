#!/bin/bash -l
#SBATCH --job-name=r1C-hetx2
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=%x-%A_%a.out
# Second grid extension for the smooth heterogeneous spectral operator at
# M = 32/64: on {0, 1e-4, ..., 1, 10} the validation error still fell at 10.
# Adds {30, 100, 1000}.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44); S=${SEEDS[$((T % 3))]}; I=$((T / 3))
MS=(32 64); CS=(32 48); WS=(30 100 1000)
M=${MS[$((I / 3))]}; CM=${CS[$((I / 3))]}; W=${WS[$((I % 3))]}
python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --epochs 4000 --batch 32 \
  --eval_every 20 --depth 4 --device cpu --hidden 192 --modes $M --cos_modes $CM \
  --seed $S --w_pde $W --out $O/routeb/smooth
