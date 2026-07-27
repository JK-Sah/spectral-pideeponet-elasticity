#!/bin/bash -l
#SBATCH --job-name=cmame-mergeB
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"
source "$PROJ/venv/bin/activate"
export CMAME_DATA="$PROJ/data"
python hetero_field_gen.py --merge --tags tr te --out "$PROJ/data/hetero_field.npz"
python -c "import numpy as np; z=np.load('$PROJ/data/hetero_field.npz'); print('MERGED', {k: z[k].shape for k in z.files})"
echo "MERGE_DONE"
