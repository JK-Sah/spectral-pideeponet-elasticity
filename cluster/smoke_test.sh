#!/bin/bash -l
#SBATCH --job-name=cmame-smoke
#SBATCH --account=flowlab
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
# End-to-end validation of the CMAME PyTorch pipeline on a GPU node:
# GPU visibility, torch CUDA, the CPU ROM baseline, and a short DeepONet train.
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$SLURM_SUBMIT_DIR"
source "$PROJ/venv/bin/activate"

echo "=== node ==="; hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=== torch device check ==="
python -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only'))"

cd "$PROJ/repo"
echo "=== ROM baseline (CPU, sanity) ==="
python rom_baseline.py --n_train 300 --n_test 100 --ranks 8 16 32 | tail -5

echo "=== random-E(x) field FEM self-test (Route B physics) ==="
python hetero_field.py --selftest | tail -5

echo "=== spectral DeepONet smoke train (GPU, 20 epochs) ==="
python continuum_elasticity_pideeponet_publication_study.py \
    --study single --epochs 20 --n_train 300 --n_test 100 \
    --true_modes 16 --n_modes_model 16 --hidden 192 --depth 4 \
    --device cuda --run_name smoke --out_dir "$PROJ/runs/smoke" 2>&1 | tail -8

echo "=== SMOKE_DONE ==="
