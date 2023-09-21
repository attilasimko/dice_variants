#!/usr/bin/env bash

for roundoff in "32"
do
for data in WMH
do
for lr in 0.0001
do
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$roundoff"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$roundoff"

for alpha1 in "-" "0"
do
for alpha2 in "-" "0"
do
for alpha3 in "-" "0"
do 
for alpha4 in "-" "0"
do 
for beta1 in "-" "0"
do
for beta2 in "-" "0"
do
for beta3 in "-" "0"
do
for beta4 in "-" "0"
do
sbatch script.sh "mime" "$lr" "$alpha1" "$alpha2" "$alpha3" "$alpha4" "$beta1" "$beta2" "$beta3" "$beta4" "$data" "$roundoff"
done
done
done
done
done
done
done
done
done
done
done