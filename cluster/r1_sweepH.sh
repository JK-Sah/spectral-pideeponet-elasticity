#!/bin/bash -l
#SBATCH --job-name=r1-swH
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-20
#SBATCH --output=%x-%A_%a.out
# Heterogeneous residual-weight sweep at the extended budget.  At 800 epochs 26
# of 27 heterogeneous runs selected an epoch at or near the end, with the
# validation error still falling, so the budget is raised to EP_HET and the
# weight is chosen again under that budget.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${EP_HET:?set EP_HET}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; W=(1e-5 1e-4 1e-3 1e-2 1e-1 1 10); S=(42 43 44)
python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --modes 16 --cos_modes 16 \
  --hidden 192 --depth 4 --epochs $EP_HET --batch 32 --eval_every 20 --device cpu \
  --w_pde ${W[$((T / 3))]} --seed ${S[$((T % 3))]} --out $PROJ/runs/r1/hetero_wsweep_e$EP_HET
