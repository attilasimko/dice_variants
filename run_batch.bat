#!/usr/bin/env bash

for data in WMH
do
for lr in 0.00001
do

sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh  "mime" "$lr" "-0.00016" "-0.07675" "-0.02461" "0.00016" "0" "0.00642" "0.00011" "-" "$data"

for alpha in "-" "0"
do
for beta in "-" "0"
do

sbatch script.sh "mime" "$lr" "0" "$alpha" "$alpha" "-" "0" "$beta" "$beta" "-" "$data"

done
done

done
done