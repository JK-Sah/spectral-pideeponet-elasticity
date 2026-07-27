#!/bin/bash -l
#SBATCH --job-name=cmame-evalN
#SBATCH --account=flowlab
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"
source "$PROJ/venv/bin/activate"
DATA="$PROJ/data/hetero_field.npz"
for CK in \
    "$PROJ/runs/route_b/fno_e_wpde0.0001_seed111.pt" \
    "$PROJ/runs/route_b/fno_e_wpde0_seed111.pt" \
    "$PROJ/runs/route_b/m32/spectral_e_wpde0.0001_seed111.pt" ; do
  echo "=== eval $CK ==="
  python eval_route_b.py --ckpt "$CK" --data "$DATA" --device cuda | tail -3
done
echo "EVALN_DONE"
