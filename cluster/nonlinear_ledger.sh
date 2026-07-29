#!/bin/bash -l
#SBATCH --job-name=cmame-nlledger
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python nonlinear_ledger.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
    --r 32 --m 128 --neural_dir "$PROJ/runs/nonlinear" --out "$PROJ/runs/nonlinear/ledger.json"
echo NLLEDGER_DONE
