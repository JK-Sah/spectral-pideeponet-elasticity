#!/bin/bash -l
#SBATCH --job-name=cmame-nlspecCPU
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=8
python nonlinear_train.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
    --model spectral --modes 64 --hidden 256 --epochs 600 --device cpu \
    --out "$PROJ/runs/nonlinear"
echo NLSPEC_CPU_DONE
