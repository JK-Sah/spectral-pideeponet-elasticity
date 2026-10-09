#!/bin/bash -l
#SBATCH --job-name=r1B-auxg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"
O=$PROJ/runs/r1/w
L=(sine_fno14 bumps_fno14 patch_fno14)
python aux_linear_v2.py --exp ${L[$SLURM_ARRAY_TASK_ID]} --w_pde $W_LIN --device cuda --data_dir $PROJ/data --out_dir $O/aux
