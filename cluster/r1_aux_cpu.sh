#!/bin/bash -l
#SBATCH --job-name=r1-aux
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
# Spectral auxiliary experiments under the canonical protocol.  EXPS is passed
# with --export so the sine and the forcing-family sets can be submitted apart.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
read -ra LIST <<< "$EXPS"
python aux_linear_v2.py --exp ${LIST[$SLURM_ARRAY_TASK_ID]} --device cpu --data_dir "$PROJ/data" --out_dir "$PROJ/runs/r1/aux"
