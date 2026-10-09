#!/bin/bash -l
#SBATCH --job-name=r1C-ling
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=%x-%A_%a.out
# Per-model weight selection for the FNOs of the manufactured benchmark.
#   0-3: canonical FNO (width 20, 12 modes), w in {0, 1e-3, 1e-2, 1e-1}
#   4-7: capacity-matched FNO (14 modes), sine family, same weights
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
T=$SLURM_ARRAY_TASK_ID; WS=(0 1e-3 1e-2 1e-1); W=${WS[$((T % 4))]}
if [ $T -lt 4 ]; then
  python canonical_linear.py --seeds 42 43 44 --trunk 16 --device auto --w_pde $W --models fno \
    --ckpt_dir $C/lin/w$W/ckpt --out $C/lin/w$W/canonical_fno_M16.json
else
  python aux_linear_v2.py --exp sine_fno14 --w_pde 0.03 --w_fno $W --device cuda \
    --data_dir $PROJ/data --out_dir $C/fno14/w$W
fi
