#!/bin/bash -l
#SBATCH --job-name=cmame-deimv2
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python pod_deim_v2.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
  --r 32 --m 64 128 192 256 --n_traj 40 --n_test 10 \
  --out "$PROJ/runs/nonlinear/deim_v2.json"
echo DEIMV2_DONE
