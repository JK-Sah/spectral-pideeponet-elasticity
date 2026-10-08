#!/bin/bash -l
#SBATCH --job-name=r1-auxg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
# Capacity-matched FNO (14 modes, ~124k parameters) on the three forcing families.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
LIST=(sine_fno14 bumps_fno14 patch_fno14)
python aux_linear_v2.py --exp ${LIST[$SLURM_ARRAY_TASK_ID]} --device cuda --data_dir "$PROJ/data" --out_dir "$PROJ/runs/r1/aux"
