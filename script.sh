#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-188 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --time=01-00:00:00
#SBATCH --error=/cephyr/users/attilas/Alvis/out/%J_error.out
#SBATCH --output=/cephyr/users/attilas/Alvis/out/%J_output.out

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source /cephyr/users/attilas/Alvis/venv/bin/activate

export var1=$1
export var2=$2
export var3=$3
python3 training.py --base alvis --optimizer "SGD" --skip_background "False" --batch_size 12 --dataset $var3 --num_epochs 200 --loss $var1 --learning_rate $var2
wait