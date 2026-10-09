#!/bin/bash -l
#SBATCH --job-name=r1-swA
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-13
#SBATCH --output=%x-%A_%a.out
# Stage A of the residual-weight re-tuning, judged on validation error only.
#   0-4 : manufactured benchmark, canonical protocol, plain PI branch
#   5-13: heterogeneous benchmark, spectral_e M=16, 3 weights x 3 seeds
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID
if [ $T -lt 5 ]; then
  W=(3e-3 3e-2 1e-1 3e-1 1)
  python ablation_canonical.py --kind wpde --value ${W[$T]} --device cpu --out "$PROJ/runs/r1/ablation/wpde_${W[$T]}.json"
else
  U=$((T - 5)); W=(1e-3 1e-2 1e-1); S=(42 43 44)
  python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --modes 16 --cos_modes 16 \
    --hidden 192 --depth 4 --epochs 800 --batch 32 --eval_every 20 --device cpu \
    --w_pde ${W[$((U / 3))]} --seed ${S[$((U % 3))]} --out $PROJ/runs/r1/hetero_wsweep
fi
