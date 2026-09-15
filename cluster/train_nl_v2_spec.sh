#!/bin/bash -l
#SBATCH --job-name=cmame-nlv2spec
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
SEEDS=(111 222 333)
python nonlinear_train_v2.py --data "$PROJ/data_nonlinear/nonlinear.npz" \
    --model spectral --modes 64 --hidden 256 --depth 4 --epochs 1000 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} --device cpu \
    --out "$PROJ/runs/nonlinear_v2"
echo NLV2_SPEC_DONE
