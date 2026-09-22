#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -A naiss2025-5-504-gpu
#SBATCH -p gpu
#SBATCH --error=/home/attilas/out/%J.err
#SBATCH --output=/home/attilas/out/%J.out

# Figures for the runs of the given training jobs, also added to each run's Comet
# experiment. submit.sh queues it to start once all of them have ended.
#
# Usage (from the repo): sbatch analyze.sh <training job id> [<training job id> ...]

set -eu
REPO="${SLURM_SUBMIT_DIR:-.}"
source "${REPO}/env.sh"

shopt -s nullglob
RUNS=()
for id in "$@"; do
    found=("${nnUNet_results}"/Dataset*/*/fold_*/*_"${id}")
    [ ${#found[@]} -eq 0 ] && echo "no run folder for job ${id}, skipping"
    RUNS+=("${found[@]}")
done
[ ${#RUNS[@]} -eq 0 ] && { echo "no runs to analyze"; exit 1; }

python "${REPO}/analyze.py" "${RUNS[@]}" --comet \
    --out "${ROOT}/dice_variants/analysis/${SLURM_JOB_ID:-local}"
