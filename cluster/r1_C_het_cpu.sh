#!/bin/bash -l
#SBATCH --job-name=r1C-het
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-47
#SBATCH --output=%x-%A_%a.out
# Per-model weight selection, heterogeneous and rough benchmarks, 4000 epochs,
# one seed per task.  w = 0.01 is covered by stage B (21799140).
#   0-17 : smooth spectral, M = 32/64/100, w in {1e-3, 1e-1}
#   18-32: rough spectral M = 64 (width 256), w in {0, 1e-4, 1e-3, 1e-1, 1}
#   33-47: anchoring test (plain and anchored), w in {0, 1e-4, 1e-3, 1e-1, 1}
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
O=$PROJ/runs/r1/w; C=$O/sel_sweep
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
T=$SLURM_ARRAY_TASK_ID; SEEDS=(42 43 44); EP=4000
COMMON="--epochs $EP --batch 32 --eval_every 20 --depth 4 --device cpu --model spectral_e"
if [ $T -lt 18 ]; then
  S=${SEEDS[$((T % 3))]}; I=$((T / 3)); MS=(32 64 100); CS=(32 48 64); WS=(1e-3 1e-1)
  M=${MS[$((I / 2))]}; CM=${CS[$((I / 2))]}; W=${WS[$((I % 2))]}
  python route_b.py --data $PROJ/data/hetero_field.npz $COMMON --hidden 192 --modes $M --cos_modes $CM \
    --seed $S --w_pde $W --out $O/routeb/smooth
elif [ $T -lt 33 ]; then
  U=$((T - 18)); S=${SEEDS[$((U % 3))]}; WS=(0 1e-4 1e-3 1e-1 1); W=${WS[$((U / 3))]}
  python route_b.py --data $PROJ/data_rough/hetero_field.npz $COMMON --hidden 256 --modes 64 --cos_modes 48 \
    --seed $S --w_pde $W --out $O/routeb/rough
else
  U=$((T - 33)); S=${SEEDS[$((U % 3))]}; WS=(0 1e-4 1e-3 1e-1 1); W=${WS[$((U / 3))]}
  python hetero_anchored.py --data $PROJ/data/hetero_field.npz --seeds $S --split_seed 42 \
    --w_pde $W --epochs $EP --device cpu --out $O/hetero/anchored_w${W}_s$S.json
fi
