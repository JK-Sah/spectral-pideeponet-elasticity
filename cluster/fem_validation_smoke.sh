#!/bin/bash -l
#SBATCH --job-name=cmame-femsmoke
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python fem_validation.py --quick --out "$PROJ/runs/nonlinear/fem_validation_smoke.json"
echo FEMVALIDATION_DONE
