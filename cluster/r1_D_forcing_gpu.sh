#!/bin/bash -l
#SBATCH --job-name=r1D-forcg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=%x-%A_%a.out
# Capacity-matched FNO on the non-sine forcing families, w in {0, 1e-3, 1e-2, 1e-1}.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
C=$PROJ/runs/r1/w/sel_sweep/forcing
T=$SLURM_ARRAY_TASK_ID; FAMS=(bumps patch); FAM=${FAMS[$((T / 4))]}; WS=(0 1e-3 1e-2 1e-1); W=${WS[$((T % 4))]}
python aux_linear_v2.py --exp ${FAM}_fno14 --w_pde 0.03 --w_fno $W --device cuda \
  --data_dir $PROJ/data --out_dir $C/w$W
