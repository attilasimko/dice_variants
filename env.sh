# Cluster environment, sourced by prepare.sh and run.sh.
# The venv's python only runs on compute nodes, not on the login node.

ROOT=/nobackup/proj/disk/naiss2025-5-504/personal/attilas
VENV_PATH=${ROOT}/nnUNet_env

export nnUNet_raw=${ROOT}/dice_variants/nnUNet_raw
export nnUNet_preprocessed=${ROOT}/dice_variants/nnUNet_preprocessed
export nnUNet_results=${ROOT}/dice_variants/nnUNet_results

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONHASHSEED=0
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${USER}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig_${USER}"

module purge
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source "${VENV_PATH}/bin/activate"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
