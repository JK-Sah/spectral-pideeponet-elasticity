#!/bin/bash -l
#SBATCH --job-name=cmame-trainR
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%a-%j.out
#SBATCH --array=0-1
# Train the surrogates on the rough field, to see whether either can stay
# accurate where the ROM's cost has cliffed.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
nvidia-smi --query-gpu=name --format=csv,noheader
DATA="$PROJ/data_rough/hetero_field.npz"; OUT="$PROJ/runs/rough"
C="--data $DATA --out $OUT --epochs 800 --batch 32 --device cuda --eval_every 25 --seed 111"
case $SLURM_ARRAY_TASK_ID in
  0) python route_b.py $C --model spectral_e --modes 64 --cos_modes 48 --hidden 256 --depth 4 --w_pde 1e-4 ;;
  1) python route_b.py $C --model fno_e --fno_width 32 --w_pde 1e-4 ;;
esac
echo "TRAINR_DONE $SLURM_ARRAY_TASK_ID"
