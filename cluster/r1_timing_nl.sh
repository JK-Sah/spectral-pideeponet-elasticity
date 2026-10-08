#!/bin/bash -l
#SBATCH --job-name=r1-tnl
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Every finite-strain timing on one exclusive allocation, four threads (same
# protocol and thread count as the linear benchmarks).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname; lscpu | grep -E "Model name" || true
python timing_nonlinear_v3.py --data $PROJ/data_nonlinear/nonlinear.npz --ckpt_dir $PROJ/runs/nonlinear_v2_long   --threads 4 --out $PROJ/runs/r1/nonlinear/timing_v3.json
