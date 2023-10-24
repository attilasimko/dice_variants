#!/usr/bin/env bash

for skip_background in "False"
do
for data in WMH
do
for lr in 0.001
do

sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"
sbatch script.sh "squared_dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$skip_background"

for alpha1 in "-" "0" "-0.0000153"
do
for beta1 in "-" "0" "0.0000076"
do
for alpha2 in "-" "0" "-0.0097581"
do 
for beta2 in "-" "0" "0.0048790"
do
for alpha3 in "-" "0" "-0.2401879"
do 
for beta3 in "-" "0" "0.1200939"
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