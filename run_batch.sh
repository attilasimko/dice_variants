#!/usr/bin/env bash
for data in ACDC WMH
do
for lr in 0.003  0.0003 0.00003
do

sbatch script.sh "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "coin" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "godl" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "dice++" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"

done
done