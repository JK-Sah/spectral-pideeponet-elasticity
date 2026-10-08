#!/bin/bash -l
#SBATCH --job-name=r1C-launch
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# After the per-model sweeps: choose every configuration's weight on
# validation, assemble the selected checkpoints and results, then submit the
# auxiliary re-runs (only where a model's weight is not 0.03), the metric
# suite and the exclusive re-timing.  Stops if any optimum is at a grid edge.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
set +e
python select_per_model.py --r1 $PROJ/runs/r1 --base $PROJ/runs/linear --assemble | tee $O/selected_per_model.txt
RC=${PIPESTATUS[0]}; set -e
if [ $RC -eq 2 ]; then echo "STAGEC_NOT_SUBMITTED: optimum at the edge of a grid"; exit 0; fi
if [ $RC -ne 0 ]; then echo "STAGEC_FAILED: select_per_model exit $RC"; exit 1; fi
eval "$(grep '^W_ANCH=' $O/selected_per_model.txt | tr ' ' '\n')"
cd $PROJ; JOBS=""
if [ "$W_ANCH" != "0.03" ]; then
  J=$(sbatch --parsable --array=0-2 --export=ALL,W_ANCH=$W_ANCH repo/cluster/r1_C_aux.sh); JOBS="$JOBS aux=$J"; fi
if [ "$W_FNO14" != "0.03" ]; then
  J=$(sbatch --parsable --export=ALL,W_FNO14=$W_FNO14 repo/cluster/r1_C_auxg.sh); JOBS="$JOBS auxg=$J"; fi
P=$(sbatch --parsable repo/cluster/r1_C_post.sh)
T=$(sbatch --parsable repo/cluster/r1_C_timing.sh)
echo "STAGEC_SUBMITTED W_ANCH=$W_ANCH W_FNO14=$W_FNO14$JOBS post=$P timing=$T"
