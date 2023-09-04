#!/bin/sh
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss dice --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss dice --skip_background True --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss cross_entropy --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss cross_entropy --skip_background True --learning_rate 0.00001

python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 1 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0.1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0.01
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0.001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0.0001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 0.1 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 0.01 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 0.001 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 0.0001 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --learning_rate 0.00001 --alpha 0 --beta 1

python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 1 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 1 --beta 0.1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 1 --beta 0.01
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 1 --beta 0.001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 1 --beta 0.0001
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 1 --beta 0
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 0.1 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 0.01 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 0.001 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 0.0001 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 200 --loss mime --skip_background True --learning_rate 0.00001 --alpha 0 --beta 1