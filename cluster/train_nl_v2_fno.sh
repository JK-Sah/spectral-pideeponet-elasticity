#!/bin/bash -l
#SBATCH --job-name=cmame-nlv2fno
#SBATCH --account=flowlab
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
SEEDS=(111 222 333)
python nonlinear_train_v2.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
    --model fno --fno_width 32 --fno_modes 12 --fno_layers 4 --epochs 1000 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} --device auto \
    --out "$PROJ/runs/nonlinear_v2"
echo NLV2_FNO_DONE
