#!/usr/bin/env bash

for skip_background in "False"
do
for data in ACDC WMH
do
for lr in 0.01
do

sbatch script.sh "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh "coin" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"

done
done
done
done
