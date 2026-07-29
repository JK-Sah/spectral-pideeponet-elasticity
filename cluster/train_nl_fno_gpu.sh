#!/bin/bash -l
#SBATCH --job-name=cmame-nlfnoGPU
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python nonlinear_train.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
    --model fno --fno_width 32 --epochs 1000 --device cuda --out "$PROJ/runs/nonlinear"
echo NLFNO_GPU_DONE
