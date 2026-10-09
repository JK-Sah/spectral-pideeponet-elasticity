#!/bin/bash -l
#SBATCH --job-name=r1B-launch
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Runs after the stage-A sweeps.  Picks each residual weight on validation error
# (select_weights.py) and submits the stage-B re-runs with those weights.  If an
# optimum lies at the edge of its sweep it stops instead, so the sweep can be
# extended first.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
python select_weights.py $PROJ/runs/r1 | tee $PROJ/runs/r1/selected_weights.txt
if grep -q WARNING $PROJ/runs/r1/selected_weights.txt; then
  echo "STAGEB_NOT_SUBMITTED: optimum at the edge of a sweep"; exit 0
fi
eval "$(grep '^W_LIN=' $PROJ/runs/r1/selected_weights.txt | tr ' ' '\n')"
cd $PROJ
EXP="--export=ALL,W_LIN=$W_LIN,W_HET=$W_HET"
mkdir -p runs/r1/w/linear/ckpt runs/r1/w/aux runs/r1/w/routeb runs/r1/w/hetero
C=$(sbatch --parsable $EXP repo/cluster/r1_B_canon.sh)
P=$(sbatch --parsable $EXP --dependency=afterok:$C repo/cluster/r1_B_post.sh)
A=$(sbatch --parsable $EXP repo/cluster/r1_B_aux.sh)
G=$(sbatch --parsable $EXP repo/cluster/r1_B_auxg.sh)
DEP="afterok:$C"
if [ "$W_HET" != "0.0001" ]; then
  HC=$(sbatch --parsable $EXP repo/cluster/r1_B_hetcpu.sh)
  HG=$(sbatch --parsable $EXP repo/cluster/r1_B_hetgpu.sh)
  DEP="$DEP:$HC:$HG"
fi
T=$(sbatch --parsable $EXP --dependency=$DEP repo/cluster/r1_B_timing.sh)
echo "STAGEB_SUBMITTED W_LIN=$W_LIN W_HET=$W_HET canon=$C post=$P aux=$A auxg=$G het=${HC:-none},${HG:-none} timing=$T"
