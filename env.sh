# Cluster environment, sourced by prepare.sh and run.sh.
# The venv's python only runs on compute nodes, not on the login node.

ROOT=/nobackup/proj/disk/naiss2025-5-504/personal/attilas
VENV_PATH=${ROOT}/nnUNet_env

export nnUNet_raw=${ROOT}/dice_variants/nnUNet_raw
export nnUNet_preprocessed=${ROOT}/dice_variants/nnUNet_preprocessed
export nnUNet_results=${ROOT}/dice_variants/nnUNet_results
# sbatch passes on the submitting shell's environment; nnUNet_n_proc_DA=0 (as set for
# the CBCT project) crashes nnU-Net's planner in torch.set_num_threads(0)
export nnUNet_n_proc_DA=${SLURM_CPUS_PER_TASK:-8}
export nnUNet_def_n_proc=${SLURM_CPUS_PER_TASK:-8}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONHASHSEED=0
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${USER}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig_${USER}"

module purge
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source "${VENV_PATH}/bin/activate"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
