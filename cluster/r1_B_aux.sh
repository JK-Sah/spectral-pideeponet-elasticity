#!/bin/bash -l
#SBATCH --job-name=r1B-aux
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-12
#SBATCH --output=%x-%A_%a.out
# Physics-informed auxiliary experiments at the validation-chosen weight
# (the data-only runs do not use the residual and are unchanged).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"
O=$PROJ/runs/r1/w
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L=(sine_nu0.3_pi sine_nu0.4_pi sine_nu0.45_pi sine_nu0.49_pi sine_nu0.499_pi sine_nu0.499_anchored sine_nu0.49_sres sine_nu0.499_sres sine_t24_pi bumps_pi bumps_anchored patch_pi patch_anchored)
python aux_linear_v2.py --exp ${L[$SLURM_ARRAY_TASK_ID]} --w_pde $W_LIN --device cpu --data_dir $PROJ/data --out_dir $O/aux
