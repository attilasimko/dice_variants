#!/usr/bin/env bash

for lr in 0.001 0.0004 0.0001 0.00004 0.00001 0.000004 0.000001 0.0000004 0.0000001
do
sbatch script.sh "dice" "$lr" 0 0 "False"
sbatch script.sh "dice" "$lr" 0 0 "True"
sbatch script.sh  "cross_entropy" "$lr" 0 0 "False"
sbatch script.sh  "cross_entropy" "$lr" 0 0 "True"

sbatch script.sh  "mime" "$lr" 1 1 "False"
sbatch script.sh  "mime" "$lr" 1 0.1 "False"
sbatch script.sh  "mime" "$lr" 1 0.01 "False"
sbatch script.sh  "mime" "$lr" 1 0.001 "False"
sbatch script.sh  "mime" "$lr" 1 0.0001 "False"
sbatch script.sh  "mime" "$lr" 1 0 "False"
sbatch script.sh  "mime" "$lr" 0.1 1 "False"
sbatch script.sh  "mime" "$lr" 0.01 1 "False"
sbatch script.sh  "mime" "$lr" 0.001 1 "False"
sbatch script.sh  "mime" "$lr" 0.0001 1 "False"
sbatch script.sh  "mime" "$lr" 0 1 "False"

sbatch script.sh  "mime" "$lr" 1 1 "True"
sbatch script.sh  "mime" "$lr" 1 0.1 "True"
sbatch script.sh  "mime" "$lr" 1 0.01 "True"
sbatch script.sh  "mime" "$lr" 1 0.001 "True"
sbatch script.sh  "mime" "$lr" 1 0.0001 "True"
sbatch script.sh  "mime" "$lr" 1 0 "True"
sbatch script.sh  "mime" "$lr" 0.1 1 "True"
sbatch script.sh  "mime" "$lr" 0.01 1 "True"
sbatch script.sh  "mime" "$lr" 0.001 1 "True"
sbatch script.sh  "mime" "$lr" 0.0001 1 "True"
sbatch script.sh  "mime" "$lr" 0 1 "True"
done