#!/bin/bash -l
#SBATCH --job-name=cmame-romB
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"
source "$PROJ/venv/bin/activate"
export CMAME_DATA="$PROJ/data"
python rom_field.py --data "$PROJ/data/hetero_field.npz" \
    --ranks 8 16 32 64 128 256 --out "$PROJ/runs/route_b/rom_field.json"
echo "ROMB_DONE"
