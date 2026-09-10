#!/bin/bash -l
#SBATCH --job-name=cmame-femval
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python fem_validation.py --out "$PROJ/runs/nonlinear/fem_validation.json"
echo FEMVALIDATION_DONE
