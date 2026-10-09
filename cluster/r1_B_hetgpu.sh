#!/bin/bash -l
#SBATCH --job-name=r1B-hetg
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"; : "${EP_HET:?set EP_HET}"
O=$PROJ/runs/r1/w
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44); S=${SEEDS[$((T % 3))]}
B="--model fno_e --fno_width 32 --epochs $EP_HET --batch 32 --eval_every 20 --device cuda --seed $S --w_pde $W_HET"
if [ $T -lt 3 ]; then python route_b.py $B --data $PROJ/data/hetero_field.npz       --out $O/routeb/smooth
else                  python route_b.py $B --data $PROJ/data_rough/hetero_field.npz --out $O/routeb/rough; fi
