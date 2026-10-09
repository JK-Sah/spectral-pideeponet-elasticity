#!/bin/bash -l
#SBATCH --job-name=r1B-hets
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=%x-%A_%a.out
# Heterogeneous / rough spectral operators and the anchoring test at the
# validation-chosen heterogeneous residual weight.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
: "${W_LIN:?set W_LIN}"; : "${W_HET:?set W_HET}"; : "${EP_HET:?set EP_HET}"
O=$PROJ/runs/r1/w
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44)
if [ $T -eq 15 ]; then
  python hetero_anchored.py --data $PROJ/data/hetero_field.npz --w_pde $W_HET --epochs $EP_HET --device cpu --out $O/hetero/anchored.json
  exit 0
fi
S=${SEEDS[$((T % 3))]}; C=$((T / 3))
SM="--data $PROJ/data/hetero_field.npz --model spectral_e --epochs $EP_HET --batch 32 --eval_every 20 --hidden 192 --depth 4 --device cpu --seed $S --w_pde $W_HET"
RO="--data $PROJ/data_rough/hetero_field.npz --model spectral_e --epochs $EP_HET --batch 32 --eval_every 20 --hidden 256 --depth 4 --device cpu --seed $S --w_pde $W_HET"
case $C in
  0) python route_b.py $SM --modes 16  --cos_modes 16 --out $O/routeb/smooth ;;
  1) python route_b.py $SM --modes 32  --cos_modes 32 --out $O/routeb/smooth ;;
  2) python route_b.py $SM --modes 64  --cos_modes 48 --out $O/routeb/smooth ;;
  3) python route_b.py $SM --modes 100 --cos_modes 64 --out $O/routeb/smooth ;;
  4) python route_b.py $RO --modes 64  --cos_modes 48 --out $O/routeb/rough  ;;
esac
