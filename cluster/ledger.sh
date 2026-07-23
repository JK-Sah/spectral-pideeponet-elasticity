#!/bin/bash -l
#SBATCH --job-name=cmame-ledger
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"
source "$PROJ/venv/bin/activate"
python ledger.py --driver --data "$PROJ/data/hetero_field.npz" \
    --out "$PROJ/runs/route_b/ledger.json"
echo "LEDGER_DONE"
