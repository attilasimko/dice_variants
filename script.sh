#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-103 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH --time=00-02:00:00
#SBATCH --mail-user=attila.simko@umu.se --mail-type=end
#SBATCH --error=/cephyr/users/attilas/Alvis/out/%J_error.out
#SBATCH --output=/cephyr/users/attilas/Alvis/out/%J_output.out

module --ignore-cache load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module --ignore-cache load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /cephyr/users/attilas/Alvis/data/newenv/newenv/bin/activate

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
python3 training.py --base alvis --optimizer "SGD" --skip_background "False" --batch_size 4 --dataset $var11 --num_epochs 200 --loss $var1 --learning_rate $var2 --alpha1 $var3 --alpha2 $var4 --alpha3 $var5 --alpha4 $var6 --beta1 $var7 --beta2 $var8 --beta3 $var9 --beta4 $var10
