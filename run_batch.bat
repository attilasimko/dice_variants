#!/usr/bin/env bash

for roundoff in "32"
do
for data in ACDC
do
for lr in 0.00001
do

sbatch script.sh "dice" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$roundoff"
sbatch script.sh  "cross_entropy" "$lr" "-" "-" "-" "-" "-" "-" "-" "-" "$data" "$roundoff"

for alpha in "-" "1" "0"
do
for beta in "-" "1" "0"
do

sbatch script.sh "mime" "$lr" "$alpha" "$alpha" "$alpha" "$alpha" "$beta" "$beta" "$beta" "$beta" "$data" "$roundoff"

done
done

done
done
done