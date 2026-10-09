#!/bin/bash -l
#SBATCH --job-name=r1C-hetg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-26
#SBATCH --output=%x-%A_%a.out
# Per-model weight selection for the heterogeneous FNO, 4000 epochs.
#   0-11 : smooth, w in {1e-4, 1e-3, 1e-1, 1}  (0 and 0.01 already run)
#   12-26: rough,  w in {0, 1e-4, 1e-3, 1e-1, 1}  (0.01 already run)
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44)
B="--model fno_e --fno_width 32 --epochs 4000 --batch 32 --eval_every 20 --device cuda"
if [ $T -lt 12 ]; then
  WS=(1e-4 1e-3 1e-1 1); W=${WS[$((T / 3))]}; S=${SEEDS[$((T % 3))]}
  python route_b.py $B --data $PROJ/data/hetero_field.npz --seed $S --w_pde $W --out $O/routeb/smooth
else
  U=$((T - 12)); WS=(0 1e-4 1e-3 1e-1 1); W=${WS[$((U / 3))]}; S=${SEEDS[$((U % 3))]}
  python route_b.py $B --data $PROJ/data_rough/hetero_field.npz --seed $S --w_pde $W --out $O/routeb/rough
fi
