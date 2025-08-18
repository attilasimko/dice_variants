#!/usr/bin/env bash
for data in ACDC WMH
do
for lr in 0.00003 0.000003
do

sbatch script.sh "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "coin" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "godl" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "dice++" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"

done
done