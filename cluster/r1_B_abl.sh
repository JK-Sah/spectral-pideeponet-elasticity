#!/bin/bash -l
#SBATCH --job-name=r1B-abl
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
# Trunk-capacity and training-set-size ablations at the validation-chosen
# residual weight.  trunk=16 and n_train=3000 are the canonical configuration,
# i.e. runs/r1/ablation/wpde_<W_LIN>.json, and are not repeated.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SET=(trunk:4 trunk:8 trunk:12 ntrain:100 ntrain:300 ntrain:1000)
KV=${SET[$SLURM_ARRAY_TASK_ID]}; K=${KV%%:*}; V=${KV#*:}
python ablation_canonical.py --kind $K --value $V --w_pde $W_LIN --device cpu \
  --out "$PROJ/runs/r1/w/ablation/${K}_${V}.json"
