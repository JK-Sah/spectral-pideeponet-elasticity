#!/bin/bash -l
#SBATCH --job-name=r1B-het0
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
# Data-only heterogeneous spectral operator (M=16) at the extended budget.  It
# does not depend on the residual weight, only on the budget.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${EP_HET:?set EP_HET}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=(42 43 44)
python route_b.py --data $PROJ/data/hetero_field.npz --model spectral_e --modes 16 --cos_modes 16 \
  --hidden 192 --depth 4 --epochs $EP_HET --batch 32 --eval_every 20 --device cpu \
  --w_pde 0 --seed ${S[$SLURM_ARRAY_TASK_ID]} --out $PROJ/runs/r1/w/routeb/smooth
