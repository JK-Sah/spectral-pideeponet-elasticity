#!/bin/bash -l
#SBATCH --job-name=cmame-trainB2
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%a-%j.out
#SBATCH --array=0-3
# Route-B v2: enlarge the sine trunk (M=16 floor was 12% on heterogeneous
# solutions; M=64 floor is 0.74%) and give the branch more capacity + richer
# E encoding. Distinct --out dirs so tags don't collide across M.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"
source "$PROJ/venv/bin/activate"
nvidia-smi --query-gpu=name --format=csv,noheader
DATA="$PROJ/data/hetero_field.npz"
OUT="$PROJ/runs/route_b"
COMMON="--data $DATA --model spectral_e --epochs 1000 --batch 32 --device cuda --eval_every 25 --seed 111 --hidden 256 --depth 4"
case $SLURM_ARRAY_TASK_ID in
  0) python route_b.py $COMMON --modes 32  --cos_modes 32 --w_pde 1e-4 --out "$OUT/m32"      ;;
  1) python route_b.py $COMMON --modes 64  --cos_modes 48 --w_pde 1e-4 --out "$OUT/m64"      ;;
  2) python route_b.py $COMMON --modes 64  --cos_modes 48 --w_pde 0    --out "$OUT/m64_data" ;;
  3) python route_b.py $COMMON --modes 100 --cos_modes 64 --w_pde 1e-4 --out "$OUT/m100"     ;;
esac
echo "TRAINB2_DONE task $SLURM_ARRAY_TASK_ID"
