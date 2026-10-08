#!/bin/bash -l
#SBATCH --job-name=r1-abl
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-13
#SBATCH --output=%x-%A_%a.out
# Fig. 5 ablations under the canonical protocol (validation selection, 3 seeds).
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SET=(wpde:1e-7 wpde:1e-6 wpde:1e-5 wpde:1e-4 wpde:1e-3 wpde:1e-2 trunk:4 trunk:8 trunk:12 trunk:16 ntrain:100 ntrain:300 ntrain:1000 ntrain:3000)
KV=${SET[$SLURM_ARRAY_TASK_ID]}; K=${KV%%:*}; V=${KV#*:}
python ablation_canonical.py --kind $K --value $V --device cpu --out "$PROJ/runs/r1/ablation/${K}_${V}.json"
