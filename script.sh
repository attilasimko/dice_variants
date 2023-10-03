#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-108 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --time=01-00:00:00
#SBATCH --error=/cephyr/users/attilas/Alvis/out/%J_error.out
#SBATCH --output=/cephyr/users/attilas/Alvis/out/%J_output.out

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source /cephyr/users/attilas/Alvis/venv/bin/activate

export var1=$1
export var2=$2
export var3=$3
export var4=$4
export var5=$5
export var6=$6
export var7=$7
export var8=$8
export var9=$9
export var10=${10}
export var11=${11}
export var12=${12}
python3 training.py --base alvis --skip_background $var12 --batch_size 12 --dataset $var11 --num_epochs 50 --loss $var1 --learning_rate $var2 --alpha1 $var3 --alpha2 $var4 --alpha3 $var5 --alpha4 $var6 --beta1 $var7 --beta2 $var8 --beta3 $var9 --beta4 $var10
wait