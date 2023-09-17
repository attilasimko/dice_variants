#!/usr/bin/env bash

for roundoff 2 4 6 8 10 12 14
do
for data in WMH ACDC
do
for lr in 0.00004
do
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data"

for alpha1 in "-"
do
for alpha2 in "-"
do
for alpha3 in "-"
do 
for alpha4 in "-"
do 
for beta1 in "-"
do
for beta2 in "-"
do
for beta3 in "-"
do
for beta4 in "-"
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