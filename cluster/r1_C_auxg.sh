#!/bin/bash -l
#SBATCH --job-name=r1C-auxg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
# Capacity-matched FNO on the non-sine families at its own weight W_FNO14.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
: "${W_FNO14:?}"
L=(bumps_fno14 patch_fno14)
python aux_linear_v2.py --exp ${L[$SLURM_ARRAY_TASK_ID]} --w_pde 0.03 --w_fno $W_FNO14 --device cuda \
  --data_dir $PROJ/data --out_dir $O/aux_sel
