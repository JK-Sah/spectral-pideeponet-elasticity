#!/bin/bash -l
#SBATCH --job-name=cmame-romR
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# POD-Galerkin ROM on the rough field, extended to high rank to expose the
# rank/cost cliff vs the per-query FEM solve.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
mkdir -p "$PROJ/runs/rough"
python rom_field.py --data "$PROJ/data_rough/hetero_field.npz" \
    --ranks 16 32 64 128 256 384 512 --out "$PROJ/runs/rough/rom_field.json"
echo "ROMR_DONE"
