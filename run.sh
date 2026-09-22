#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2025-5-504-gpu
#SBATCH -p gpu
#SBATCH --error=/home/attilas/out/%J.err
#SBATCH --output=/home/attilas/out/%J.out

# Usage (from the repo): sbatch run.sh <ACDC|WMH> <dice|ce|dice_ce> [seed] [train.py options]
#
#   sbatch run.sh WMH dice
#   sbatch run.sh ACDC dice 1 --momentum 0 --lr 1

set -eu
USAGE="Usage: sbatch run.sh <ACDC|WMH> <dice|ce|dice_ce> [seed] [train.py options]"
DATASET="${1:?${USAGE}}"
LOSS="${2:?${USAGE}}"
REPO="${SLURM_SUBMIT_DIR:-.}"
source "${REPO}/env.sh"
python "${REPO}/train.py" --dataset "${DATASET}" --loss "${LOSS}" --seed "${3:-0}" "${@:4}"
