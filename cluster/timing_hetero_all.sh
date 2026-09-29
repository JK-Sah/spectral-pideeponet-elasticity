#!/bin/bash -l
#SBATCH --job-name=cmame-thet
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
hostname; lscpu | grep -E "Model name" || true

echo; echo "################ SMOOTH FIELD ################"
python timing_hetero_all.py --data "$PROJ/data/hetero_field.npz" \
  --ckpt_dir "$PROJ/runs/route_b" --ranks 8 16 32 64 128 256 \
  --out "$PROJ/runs/hetero/timing_smooth.json"

echo; echo "################ ROUGH FIELD ################"
python timing_hetero_all.py --data "$PROJ/data_rough/hetero_field.npz" \
  --ckpt_dir "$PROJ/runs/rough" --ranks 16 32 64 128 256 384 512 \
  --out "$PROJ/runs/rough/timing_rough.json"
echo TIMINGHETERO_ALL_DONE
