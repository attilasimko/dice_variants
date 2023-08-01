#!/bin/sh
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss cross_entropy --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss dice --learning_rate 0.00001
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss mime --learning_rate 0.00001

python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss cross_entropy --learning_rate 0.00005
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss dice --learning_rate 0.00005
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss mime --learning_rate 0.00005

python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss cross_entropy --learning_rate 0.0001
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss dice --learning_rate 0.0001
python3 training.py --gpu 2 --base gauss --num_epochs 75 --loss mime --learning_rate 0.0001
