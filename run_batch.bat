#!/usr/bin/env bash

for skip_background in "False"
do
for data in WMH
do
for lr in 0.001
do

# sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
# sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
# sbatch script.sh "squared_dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"

for alpha1 in "-"
do
for beta1 in "-"
do
for alpha2 in "-"
do 
for beta2 in "-"
do
for alpha3 in "-"
do 
for beta3 in "-"
do

sbatch script.sh "coin" "$lr" "$alpha1" "$alpha2" "$alpha3" "-" "$beta1" "$beta2" "$beta3" "-" "$data" "$skip_background"

done
done
done
done

done
done
done
done
done