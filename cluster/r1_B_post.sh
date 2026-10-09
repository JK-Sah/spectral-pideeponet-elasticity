#!/bin/bash -l
#SBATCH --job-name=r1B-post
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Metric suite, noise/OOD and field snapshots from the new canonical checkpoints.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"
O=$PROJ/runs/r1/w
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cp -n $PROJ/runs/linear/ckpt/data_only_spectral_M*_seed*.pt $O/linear/ckpt/
for M in 16 64; do
  python canonical_metrics_all.py --ckpt_dir $O/linear/ckpt --trunk $M --out $O/linear/canonical_metrics_all_M${M}.json
done
python eval_canonical_extras.py --ckpt_dir $O/linear/ckpt --out $O/linear/canonical_extras.json
