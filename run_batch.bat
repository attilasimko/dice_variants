#!/usr/bin/env bash

for lr in 0.00004
do
sbatch script.sh "dice" "$lr" 0 0 "0" "0" "0" "0" "0" "0"
# sbatch script.sh  "cross_entropy" "$lr" 0 0 "0" "0" "0" "0" "0" "0"

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
sbatch script.sh  "mime" "$lr" "$alpha1" "$alpha2" "$alpha3" "$beta1" "$beta2" "$beta3"
done
done
done
done
done
done
done