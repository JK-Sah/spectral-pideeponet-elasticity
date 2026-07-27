#!/bin/bash -l
#SBATCH --job-name=cmame-trainB
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%a-%j.out
#SBATCH --array=0-3
# Route-B training: hero (spectral+PDE), data-only ablation, FNO baseline +/- PDE.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"
source "$PROJ/venv/bin/activate"
nvidia-smi --query-gpu=name --format=csv,noheader
DATA="$PROJ/data/hetero_field.npz"
OUT="$PROJ/runs/route_b"
COMMON="--data $DATA --out $OUT --epochs 800 --batch 32 --device cuda --eval_every 25 --seed 111"
case $SLURM_ARRAY_TASK_ID in
  0) python route_b.py --model spectral_e --w_pde 1e-4 $COMMON ;;
  1) python route_b.py --model spectral_e --w_pde 0    $COMMON ;;
  2) python route_b.py --model fno_e      --w_pde 0    $COMMON ;;
  3) python route_b.py --model fno_e      --w_pde 1e-4 $COMMON ;;
esac
echo "TRAINB_DONE task $SLURM_ARRAY_TASK_ID"
