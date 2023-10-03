#!/usr/bin/env bash

for skip_background in "True" "False"
do
for data in WMH
do
for lr in 0.01 0.005 0.001 0.0005 0.0001
do

sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
sbatch script.sh  "coin" "$lr" "-0.00098" "-0.15493" "-0.06960" "-" "0.00049" "0.00432" "0.00009" "-" "$data" "$skip_background"

for alpha in "-"
do
for beta in "-"
do

sbatch script.sh "coin" "$lr" "$alpha" "$alpha" "$alpha" "-" "$beta" "$beta" "$beta" "-" "$data" "$skip_background"

done
done

done
done
done