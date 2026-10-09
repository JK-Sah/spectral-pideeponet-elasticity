#!/bin/bash -l
#SBATCH --job-name=r1B-canon
#SBATCH --account=flowlab
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
# Canonical configuration at the validation-chosen residual weight, M = 16/32/64.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"
O=$PROJ/runs/r1/w
M=(16 32 64); M=${M[$SLURM_ARRAY_TASK_ID]}
python canonical_linear.py --seeds 42 43 44 --trunk $M --device auto --w_pde $W_LIN \
  --models pi_spectral_plain pi_spectral_anchored fno \
  --ckpt_dir $O/linear/ckpt --out $O/linear/canonical_M${M}.json
