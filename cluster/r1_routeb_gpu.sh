#!/bin/bash -l
#SBATCH --job-name=r1-rbg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=%x-%A_%a.out
# Heterogeneous and rough-field FNOs, validation-selected, 3 seeds.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44); S=${SEEDS[$((T % 3))]}; C=$((T / 3))
B="--model fno_e --fno_width 32 --epochs 800 --batch 32 --eval_every 20 --device cuda --seed $S"
O=$PROJ/runs/r1/routeb
case $C in
  0) python route_b.py $B --data $PROJ/data/hetero_field.npz       --w_pde 1e-4 --out $O/smooth ;;
  1) python route_b.py $B --data $PROJ/data/hetero_field.npz       --w_pde 0    --out $O/smooth ;;
  2) python route_b.py $B --data $PROJ/data_rough/hetero_field.npz --w_pde 1e-4 --out $O/rough  ;;
esac
