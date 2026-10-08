#!/bin/bash -l
#SBATCH --job-name=r1B-launch2
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Heterogeneous half of stage B.  The linear half was submitted by hand with
# W_LIN=0.03 (interior optimum of stage A); this runs after the extended
# heterogeneous sweep, picks W_HET on validation error, and submits the
# heterogeneous re-runs and the exclusive re-timing.  CANON is the job id of
# the stage-B canonical array, which the re-timing also waits for.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
: "${CANON:?set CANON}"
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python select_weights.py $PROJ/runs/r1 | tee $PROJ/runs/r1/selected_weights.txt
if grep -q WARNING $PROJ/runs/r1/selected_weights.txt; then
  echo "STAGEB_NOT_SUBMITTED: optimum at the edge of a sweep"; exit 0
fi
eval "$(grep '^W_LIN=' $PROJ/runs/r1/selected_weights.txt | tr ' ' '\n')"
cd $PROJ
EXP="--export=ALL,W_LIN=$W_LIN,W_HET=$W_HET"
mkdir -p runs/r1/w/routeb runs/r1/w/hetero
HC=$(sbatch --parsable $EXP repo/cluster/r1_B_hetcpu.sh)
HG=$(sbatch --parsable $EXP repo/cluster/r1_B_hetgpu.sh)
DEP="afterok:$HC:$HG"
# still queued or running: wait for it; already finished: the record may be gone
if squeue -h -j "$CANON" 2>/dev/null | grep -q .; then DEP="$DEP:$CANON"; fi
T=$(sbatch --parsable $EXP --dependency=$DEP repo/cluster/r1_B_timing.sh)
echo "STAGEB_HET_SUBMITTED W_LIN=$W_LIN W_HET=$W_HET het=$HC,$HG timing=$T (dep $DEP)"
