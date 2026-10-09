#!/bin/bash -l
#SBATCH --job-name=r1C-time
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Exclusive four-thread re-timing of the per-model selected checkpoints,
# linear and heterogeneous, so each ledger's accuracy and cost refer to the
# same trained weights.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname; lscpu | grep -E "Model name" || true
for M in 16 64; do
  python timing_linear_v3.py --ckpt_dir $O/linear_sel/ckpt --trunk $M --threads 4 --order blocked \
    --out $O/linear_sel/timing_linear_v3_M${M}.json
done
python timing_hetero_all.py --data $PROJ/data/hetero_field.npz --ckpt_dir $O/routeb/smooth_sel \
  --ranks 8 16 32 64 128 256 --out $O/routeb/timing_smooth.json
python timing_hetero_all.py --data $PROJ/data_rough/hetero_field.npz --ckpt_dir $O/routeb/rough_sel \
  --ranks 16 32 64 128 256 384 512 --out $O/routeb/timing_rough.json
