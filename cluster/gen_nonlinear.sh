#!/bin/bash -l
#SBATCH --job-name=cmame-genNL
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%a-%j.out
#SBATCH --array=0-19
# Finite-strain hyperelastic ground truth (Newton FEM, ~3 s/sample).
# Tasks 0-15 = train (2000, 125 each); tasks 16-19 = test (400, 100 each).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export CMAME_DATA="$PROJ/data_nonlinear"
TID=$SLURM_ARRAY_TASK_ID
if [ "$TID" -lt 16 ]; then
  S=$((TID*125)); E=$((S+125))
  python nonlinear_gen.py --tag tr --n 2000 --seed 42  --start $S --stop $E --scale 10
else
  J=$((TID-16)); S=$((J*100)); E=$((S+100))
  python nonlinear_gen.py --tag te --n 400  --seed 999 --start $S --stop $E --scale 10
fi
echo "GENNL_DONE $TID"
