#!/bin/bash -l
#SBATCH --job-name=cmame-genR
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%a-%j.out
#SBATCH --array=0-9
# Rough (short-correlation, high n-width) E(x) ground truth for the crossover test.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export CMAME_DATA="$PROJ/data_rough"
TID=$SLURM_ARRAY_TASK_ID
if [ "$TID" -lt 8 ]; then
  S=$((TID*250)); E=$((S+250))
  python hetero_field_gen.py --tag tr --n 2000 --seed 42  --start $S --stop $E --refine 4 --corr 1.0
else
  J=$((TID-8)); S=$((J*200)); E=$((S+200))
  python hetero_field_gen.py --tag te --n 400  --seed 999 --start $S --stop $E --refine 4 --corr 1.0
fi
echo "GENR_DONE $TID"
