#!/bin/bash -l
#SBATCH --job-name=cmame-mergeNL
#SBATCH --account=flowlab
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
set -euo pipefail
PROJ=/shared/rc/whiskers/JK/cmame_pideeponet
cd "$PROJ/repo"; source "$PROJ/venv/bin/activate"
export CMAME_DATA="$PROJ/data_nonlinear"
python nonlinear_gen.py --merge --tags tr te --out "$PROJ/data_nonlinear/nonlinear.npz"
python -c "import numpy as np; z=np.load('$PROJ/data_nonlinear/nonlinear.npz'); print('MERGED',{k:z[k].shape for k in z.files}); print('peak|u| mean %.3f max %.3f'%(np.abs(z['u_tr']).max(axis=(1,2,3)).mean(), np.abs(z['u_tr']).max()))"
echo "MERGENL_DONE"
