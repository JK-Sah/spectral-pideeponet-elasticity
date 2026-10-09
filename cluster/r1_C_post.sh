#!/bin/bash -l
#SBATCH --job-name=r1C-post
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Metric suite, noise/OOD and field snapshots from the per-model selected checkpoints.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for M in 16 64; do
  python canonical_metrics_all.py --ckpt_dir $O/linear_sel/ckpt --trunk $M --out $O/linear_sel/canonical_metrics_all_M${M}.json
done
python eval_canonical_extras.py --ckpt_dir $O/linear_sel/ckpt --out $O/linear_sel/canonical_extras.json
