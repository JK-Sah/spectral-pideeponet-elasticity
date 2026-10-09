#!/bin/bash -l
#SBATCH --job-name=r1C-aux
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
# Auxiliary experiments of the anchored branch at its own validation-chosen
# weight W_ANCH (array index into the list below; the launcher passes --array).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
: "${W_ANCH:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L=(sine_nu0.499_anchored bumps_anchored patch_anchored)
python aux_linear_v2.py --exp ${L[$SLURM_ARRAY_TASK_ID]} --w_pde 0.03 --w_anchored $W_ANCH --device cpu \
  --data_dir $PROJ/data --out_dir $O/aux_sel
