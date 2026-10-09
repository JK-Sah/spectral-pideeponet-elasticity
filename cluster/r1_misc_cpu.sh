#!/bin/bash -l
#SBATCH --job-name=r1-misc
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=%x-%A_%a.out
# FEM refinement on both modulus fields; finite-strain classical baselines on the
# full test set; noise/OOD/field snapshots from the canonical checkpoints.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
O=$PROJ/runs/r1
case $SLURM_ARRAY_TASK_ID in
  0) python hetero_fem_refine.py --data $PROJ/data/hetero_field.npz       --corr 6.0 --out $O/refine/smooth.json ;;
  1) python hetero_fem_refine.py --data $PROJ/data_rough/hetero_field.npz --corr 1.0 --out $O/refine/rough.json ;;
  2) python nonlinear_classical_full.py --data $PROJ/data_nonlinear/nonlinear.npz --out $O/nonlinear/classical_full.json ;;
  3) python eval_canonical_extras.py --ckpt_dir $PROJ/runs/linear/ckpt --out $O/linear/canonical_extras.json ;;
esac
