#!/bin/bash -l
#SBATCH --job-name=r1-rbs
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-17
#SBATCH --output=%x-%A_%a.out
# Heterogeneous and rough-field spectral operators, validation-selected, 3 seeds.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44); S=${SEEDS[$((T % 3))]}; C=$((T / 3))
SM="--data $PROJ/data/hetero_field.npz --model spectral_e --epochs 800 --batch 32 --eval_every 20 --hidden 192 --depth 4 --device cpu --seed $S"
RO="--data $PROJ/data_rough/hetero_field.npz --model spectral_e --epochs 800 --batch 32 --eval_every 20 --hidden 256 --depth 4 --device cpu --seed $S"
O=$PROJ/runs/r1/routeb
case $C in
  0) python route_b.py $SM --modes 16  --cos_modes 16 --w_pde 1e-4 --out $O/smooth ;;
  1) python route_b.py $SM --modes 16  --cos_modes 16 --w_pde 0    --out $O/smooth ;;
  2) python route_b.py $SM --modes 32  --cos_modes 32 --w_pde 1e-4 --out $O/smooth ;;
  3) python route_b.py $SM --modes 64  --cos_modes 48 --w_pde 1e-4 --out $O/smooth ;;
  4) python route_b.py $SM --modes 100 --cos_modes 64 --w_pde 1e-4 --out $O/smooth ;;
  5) python route_b.py $RO --modes 64  --cos_modes 48 --w_pde 1e-4 --out $O/rough  ;;
esac
