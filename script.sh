#!/bin/sh
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 1 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 1 --beta 0
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 0 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 2 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 2 --beta 0
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 0 --beta 2
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 10 --beta 1
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 10 --beta 0
python3 training.py --gpu 2 --base gauss --num_epochs 150 --loss mime --learning_rate 0.00001 --alpha 0 --beta 10