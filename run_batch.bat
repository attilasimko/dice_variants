#!/usr/bin/env bash

for data in WMH ACDC
do
for lr in 0.0004 0.0001 0.00004 0.00001 0.000004
do
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"

for alpha1 in "-"
do
for alpha2 in "-"
do
for alpha3 in "-"
do 
for beta1 in "-"
do
for beta2 in "-"
do
for beta3 in "-"
do
sbatch script.sh  "mime" "$lr" "$alpha1" "$alpha2" "$alpha3" "$alpha4" "$beta1" "$beta2" "$beta3" "$beta4" "$data"
done
done
done
done
done
done
done
done