#!/bin/sh
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss cross_entropy --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss dice --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 1 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0.01
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0.0001
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 0.01 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 0.0001 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 0 --beta 1