#!/usr/bin/env bash

for skip_background in "True"
do
for data in WMH
do
for lr in 0.0005
do

sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"

for alpha2 in "rand"
do 
for beta2 in "rand"
do
for alpha3 in "rand"
do 
for beta3 in "rand"
do

sbatch script.sh "coin" "$lr" "-" "$alpha2" "$alpha3" "-" "-" "$beta2" "$beta3" "-" "$data" "$skip_background"

done
done
done
done

done
done
done