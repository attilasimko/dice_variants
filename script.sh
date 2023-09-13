#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-108 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --time=00-12:00:00
#SBATCH --mail-user=attila.simko@umu.se --mail-type=end
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
python3 training.py --base alvis --num_epochs 120 --loss $var1 --learning_rate $var2 --alpha1 $var3 --alpha2 $var4 --alpha3 $var5 --beta1 $var6 --beta2 $var7 --beta3 $var8
wait