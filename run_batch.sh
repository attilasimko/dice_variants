#!/usr/bin/env bash
for data in WMH
do
for lr in 0.0001
do

sbatch script.sh "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "coin" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"

done
done