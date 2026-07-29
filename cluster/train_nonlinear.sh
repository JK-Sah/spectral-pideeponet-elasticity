#!/bin/bash -l
#SBATCH --job-name=cmame-trainNL
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%a-%j.out
#SBATCH --array=0-1
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
nvidia-smi --query-gpu=name --format=csv,noheader
DATA="$PROJ/data_nonlinear/nonlinear.npz"; OUT="$PROJ/runs/nonlinear"
case $SLURM_ARRAY_TASK_ID in
  0) python nonlinear_train.py --data "$DATA" --model spectral --modes 64 --hidden 256 --epochs 1000 --device cuda --out "$OUT" ;;
  1) python nonlinear_train.py --data "$DATA" --model fno --fno_width 32 --epochs 1000 --device cuda --out "$OUT" ;;
esac
echo "TRAINNL_DONE $SLURM_ARRAY_TASK_ID"
