#!/bin/bash -l
#SBATCH --job-name=cmame-tclass
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python timing_classical_linear.py --threads 4 \
  --out "$PROJ/runs/linear/timing_classical.json"
echo TIMINGCLASSICAL_DONE
