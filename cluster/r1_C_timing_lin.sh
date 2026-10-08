#!/bin/bash -l
#SBATCH --job-name=r1C-tlin
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Linear benchmark: classical baselines and the per-model selected learned
# checkpoints timed in ONE exclusive allocation, as timing_all.sh did for the
# earlier checkpoints.  The stage-C re-timing (21801951, skl-a-24) timed only
# the learned models, and its single-query latencies ran about 45% above the
# earlier node's for identical models (closed form 1.02 -> 1.49 ms), so they
# cannot be set beside classical times measured elsewhere.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w/linear_sel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "### node"; hostname
lscpu | grep -E "Model name|Socket|Thread|CPU max" || true
echo; echo "### classical baselines, both benchmarks, one protocol"
python timing_classical_all.py --threads 4 --het_data "$PROJ/data/hetero_field.npz" \
  --out "$O/timing_classical_all.json"
for M in 16 64; do
  echo; echo "### learned models, M=$M, all protocol conditions"
  python timing_linear_v3.py --ckpt_dir "$O/ckpt" --trunk $M --threads 4 --order both \
    --out "$O/timing_linear_v3_M${M}.json"
done
echo TIMINGALL_DONE
