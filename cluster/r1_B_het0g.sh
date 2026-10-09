#!/bin/bash -l
#SBATCH --job-name=r1B-het0g
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
# Data-only heterogeneous FNO at the extended budget.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${EP_HET:?set EP_HET}"
S=(42 43 44)
python route_b.py --data $PROJ/data/hetero_field.npz --model fno_e --fno_width 32 --epochs $EP_HET \
  --batch 32 --eval_every 20 --device cuda --seed ${S[$SLURM_ARRAY_TASK_ID]} --w_pde 0 \
  --out $PROJ/runs/r1/w/routeb/smooth
