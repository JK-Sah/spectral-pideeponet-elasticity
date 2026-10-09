#!/bin/bash -l
#SBATCH --job-name=r1B-time
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Re-time the new linear and heterogeneous checkpoints on one exclusive
# four-thread allocation (same protocol as before), so every ledger's accuracy
# and cost refer to the same trained weights.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"
O=$PROJ/runs/r1/w
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname; lscpu | grep -E "Model name" || true
cp -n $PROJ/runs/linear/ckpt/data_only_spectral_M*_seed*.pt $O/linear/ckpt/
for M in 16 64; do
  python timing_linear_v3.py --ckpt_dir $O/linear/ckpt --trunk $M --threads 4 --order blocked \
    --out $O/linear/timing_linear_v3_M${M}.json
done
if [ "$W_HET" = "0.0001" ]; then echo "heterogeneous weight unchanged: timed by job r1-thet"; exit 0; fi
mkdir -p $O/routeb/smooth_all $O/routeb/rough_all
cp -n $O/routeb/smooth/*.pt $O/routeb/smooth_all/ 2>/dev/null || true   # includes the data-only runs
cp -n $O/routeb/rough/*.pt $O/routeb/rough_all/ 2>/dev/null || true
python timing_hetero_all.py --data $PROJ/data/hetero_field.npz --ckpt_dir $O/routeb/smooth_all \
  --ranks 8 16 32 64 128 256 --out $O/routeb/timing_smooth.json
python timing_hetero_all.py --data $PROJ/data_rough/hetero_field.npz --ckpt_dir $O/routeb/rough_all \
  --ranks 16 32 64 128 256 384 512 --out $O/routeb/timing_rough.json
