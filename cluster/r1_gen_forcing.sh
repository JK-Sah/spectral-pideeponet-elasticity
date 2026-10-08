#!/bin/bash -l
#SBATCH --job-name=r1-genF
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
# Non-sine forcing families at canonical sizes + FEM/ROM/closed-form rows.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
KINDS=(bumps patch)
python gen_forcing_v2.py --kind ${KINDS[$SLURM_ARRAY_TASK_ID]} --out_dir "$PROJ/data" --res_out "$PROJ/runs/r1/aux"
