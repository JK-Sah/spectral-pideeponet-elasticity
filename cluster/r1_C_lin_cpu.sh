#!/bin/bash -l
#SBATCH --job-name=r1C-lin
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-17
#SBATCH --output=%x-%A_%a.out
# Per-model residual-weight selection, manufactured benchmark, canonical
# protocol.  1e-4 (runs/linear) and 0.03 (stage B) already exist for every
# configuration; this adds the rest of the grid.
#   0-11 : anchored branch, M = 16/32/64, w in {0, 1e-3, 1e-2, 1e-1}
#   12-17: plain branch,    M = 32/64,    w in {1e-3, 1e-2, 1e-1}
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID
if [ $T -lt 12 ]; then
  MS=(16 32 64); WS=(0 1e-3 1e-2 1e-1); M=${MS[$((T / 4))]}; W=${WS[$((T % 4))]}; TAG=pi_spectral_anchored
else
  U=$((T - 12)); MS=(32 64); WS=(1e-3 1e-2 1e-1); M=${MS[$((U / 3))]}; W=${WS[$((U % 3))]}; TAG=pi_spectral_plain
fi
python canonical_linear.py --seeds 42 43 44 --trunk $M --device cpu --w_pde $W --models $TAG \
  --ckpt_dir $C/lin/w$W/ckpt --out $C/lin/w$W/canonical_${TAG}_M$M.json
