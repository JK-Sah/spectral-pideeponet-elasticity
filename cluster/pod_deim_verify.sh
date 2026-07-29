#!/bin/bash -l
#SBATCH --job-name=cmame-deimA
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python pod_deim.py --selftest --data "$PROJ/data_nonlinear/nonlinear.npz"
echo DEIMA_VERIFY_DONE
