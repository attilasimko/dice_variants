#!/usr/bin/env bash
# Submits the Dice runs (ACDC and WMH; nnU-Net's momentum 0.99, and momentum 0 with
# lr 1 for per-sample attribution) and one analysis job that starts when all of them
# have ended, whether they finished or not.
#
# Usage (on the login node, from the repo): bash submit.sh [train.py options for all runs]

set -eu
IDS=()
for d in ACDC WMH; do
    for opts in "" "--momentum 0 --lr 1"; do
        # shellcheck disable=SC2086  # opts is deliberately word-split
        id=$(sbatch --parsable run.sh "${d}" dice 0 ${opts} "$@")
        IDS+=("${id%%;*}")
        echo "${d} dice ${opts:-(nnU-Net default)}: job ${id%%;*}"
    done
done
DEPS=$(IFS=:; echo "${IDS[*]}")
id=$(sbatch --parsable --dependency=afterany:"${DEPS}" analyze.sh "${IDS[@]}")
echo "analysis: job ${id%%;*}, after ${DEPS}"
