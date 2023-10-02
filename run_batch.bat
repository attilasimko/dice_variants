#!/usr/bin/env bash

for skip_background in "True"
do
for data in WMH
do
for lr in 0.001 0.0005 0.0001 0.00005 0.00001 0.000005 0.000001
do

sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
# sbatch script.sh  "mime" "$lr" "-0.00016" "-0.07675" "-0.02461" "0.00016" "0" "0.00642" "0.00011" "-" "$data" "$skip_background"

for alpha in "-"
do
for beta in "-"
do

sbatch script.sh "mime" "$lr" "$alpha" "$alpha" "$alpha" "-" "$beta" "$beta" "$beta" "-" "$data" "$skip_background"

done
done

done
done
done