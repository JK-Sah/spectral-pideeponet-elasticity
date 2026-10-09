#!/bin/bash -l
#SBATCH --job-name=r1-thet
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Re-time the validation-selected heterogeneous checkpoints, classical and
# learned, on one exclusive four-thread allocation.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname; lscpu | grep -E "Model name" || true
python timing_hetero_all.py --data $PROJ/data/hetero_field.npz --ckpt_dir $PROJ/runs/r1/routeb/smooth   --ranks 8 16 32 64 128 256 --out $PROJ/runs/r1/routeb/timing_smooth.json
python timing_hetero_all.py --data $PROJ/data_rough/hetero_field.npz --ckpt_dir $PROJ/runs/r1/routeb/rough   --ranks 16 32 64 128 256 384 512 --out $PROJ/runs/r1/routeb/timing_rough.json
