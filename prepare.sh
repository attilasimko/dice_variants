#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gpus 1
#SBATCH -t 04:00:00
#SBATCH -A naiss2025-5-504-gpu
#SBATCH -p gpu
#SBATCH --error=/home/attilas/out/%J.err
#SBATCH --output=/home/attilas/out/%J.out

# One-off: raw ACDC + WMH -> nnU-Net datasets 027/028 with patient-level splits,
# then nnU-Net planning and preprocessing (2d and 3d_fullres).
#
# Usage (from the repo): sbatch prepare.sh [acdc_dir] [wmh_dir]

set -eu
ROOT=/nobackup/proj/disk/naiss2025-5-504/personal/attilas
ACDC_DIR="${1:-${ROOT}/ACDC}"
WMH_DIR="${2:-${ROOT}/WMH}"
REPO="${SLURM_SUBMIT_DIR:-.}"

source "${REPO}/env.sh"
python -c "import nnunetv2, importlib.metadata as m; print('nnunetv2', m.version('nnunetv2'))"

python "${REPO}/convert.py" --acdc "${ACDC_DIR}" --wmh "${WMH_DIR}"
nnUNetv2_plan_and_preprocess -d 27 28 -c 2d 3d_fullres -np 8 8 --verify_dataset_integrity
