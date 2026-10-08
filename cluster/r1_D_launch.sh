#!/bin/bash -l
#SBATCH --job-name=r1D-launch
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# After the forcing-family sweeps: choose each model's weight on each family's
# validation set and copy the selected results into r1/w/aux_sel.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
set +e
python select_forcing.py --r1 $PROJ/runs/r1 --assemble | tee $PROJ/runs/r1/w/forcing_selected.txt
RC=${PIPESTATUS[0]}; set -e
if [ $RC -eq 2 ]; then echo "FORCING_NOT_ASSEMBLED: optimum at the edge of a grid"; exit 0; fi
[ $RC -eq 0 ] && echo "FORCING_SELECTED" || { echo "FORCING_FAILED $RC"; exit 1; }
