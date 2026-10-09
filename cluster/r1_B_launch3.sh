#!/bin/bash -l
#SBATCH --job-name=r1B-launch3
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Heterogeneous half of stage B at the extended budget EP_HET.  Runs after the
# EP_HET weight sweep, picks W_HET on validation error, submits the
# heterogeneous re-runs and then the exclusive re-timing, which also waits for
# the data-only runs (HET0, HET0G) and, if still queued, the canonical array.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
: "${EP_HET:?}"; : "${HET0:?}"; : "${HET0G:?}"; : "${CANON:?}"
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python select_weights.py $PROJ/runs/r1 --het_dir hetero_wsweep_e$EP_HET | tee $PROJ/runs/r1/selected_weights_e$EP_HET.txt
if grep -q WARNING $PROJ/runs/r1/selected_weights_e$EP_HET.txt; then
  echo "STAGEB_NOT_SUBMITTED: optimum at the edge of a sweep"; exit 0
fi
eval "$(grep '^W_LIN=' $PROJ/runs/r1/selected_weights_e$EP_HET.txt | tr ' ' '\n')"
cd $PROJ
EXP="--export=ALL,W_LIN=$W_LIN,W_HET=$W_HET,EP_HET=$EP_HET"
HC=$(sbatch --parsable $EXP repo/cluster/r1_B_hetcpu.sh)
HG=$(sbatch --parsable $EXP repo/cluster/r1_B_hetgpu.sh)
DEP="afterok:$HC:$HG"
for J in $HET0 $HET0G $CANON; do
  if squeue -h -j "$J" 2>/dev/null | grep -q .; then DEP="$DEP:$J"; fi
done
T=$(sbatch --parsable $EXP --dependency=$DEP repo/cluster/r1_B_timing.sh)
echo "STAGEB_HET_SUBMITTED EP_HET=$EP_HET W_LIN=$W_LIN W_HET=$W_HET het=$HC,$HG timing=$T (dep $DEP)"
