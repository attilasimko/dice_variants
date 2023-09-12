#!/usr/bin/env bash

for lr in 0.00004
do
sbatch script.sh "dice" "$lr" 0 0 "0" "0" "0" "0" "0" "0" "False"
sbatch script.sh "dice" "$lr" 0 0 "0" "0" "0" "0" "0" "0" "False"
# sbatch script.sh  "cross_entropy" "$lr" 0 0 "0" "0" "0" "0" "0" "0"  "False"
sbatch script.sh  "mime" "$lr" "0" "0" "0" "0" "0" "0" "True"

# for alpha1 in 0 1e6
# do
# for alpha2 in 0 1e6
# do
# for alpha3 in 0 1e6
# do 
# for beta1 in 0 1e-4
# do
# for beta2 in 0 1e-4
# do
# for beta3 in 0 1e-4
# do
# sbatch script.sh  "mime" "$lr" "$alpha1" "$alpha2" "$alpha3" "$beta1" "$beta2" "$beta3" "False"
# done
# done
# done
# done
# done
# done
done