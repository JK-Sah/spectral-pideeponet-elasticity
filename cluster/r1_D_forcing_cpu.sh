#!/bin/bash -l
#SBATCH --job-name=r1D-forc
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-13
#SBATCH --output=%x-%A_%a.out
# Per-family residual-weight selection on the non-sine forcing families: the
# grid points not yet run.  1e-4 and 0.03 exist (r1/aux, r1/w/aux); the plain
# branch at 0 is the data-only run.
#   per family (bumps, patch): plain {1e-3, 1e-2, 1e-1}, anchored {0, 1e-3, 1e-2, 1e-1}
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
C=$PROJ/runs/r1/w/sel_sweep/forcing
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; FAMS=(bumps patch); FAM=${FAMS[$((T / 7))]}; U=$((T % 7))
if [ $U -lt 3 ]; then
  WS=(1e-3 1e-2 1e-1); W=${WS[$U]}
  python aux_linear_v2.py --exp ${FAM}_pi --w_pde $W --device cpu --data_dir $PROJ/data --out_dir $C/w$W
else
  WS=(0 1e-3 1e-2 1e-1); W=${WS[$((U - 3))]}
  python aux_linear_v2.py --exp ${FAM}_anchored --w_pde 0.03 --w_anchored $W --device cpu \
    --data_dir $PROJ/data --out_dir $C/w$W
fi
