#!/bin/sh
python training.py --gpu 0 --base gauss --num_epochs 30 --loss dice --learning_rate 0.00001
python training.py --gpu 0 --base gauss --num_epochs 30 --loss cross_entropy --learning_rate 0.00001
python training.py --gpu 0 --base gauss --num_epochs 30 --loss mime --learning_rate 0.00001

python training.py --gpu 0 --base gauss --num_epochs 30 --loss dice --learning_rate 0.00005
python training.py --gpu 0 --base gauss --num_epochs 30 --loss cross_entropy --learning_rate 0.00005
python training.py --gpu 0 --base gauss --num_epochs 30 --loss mime --learning_rate 0.00005

python training.py --gpu 0 --base gauss --num_epochs 30 --loss dice --learning_rate 0.0001
python training.py --gpu 0 --base gauss --num_epochs 30 --loss cross_entropy --learning_rate 0.0001
python training.py --gpu 0 --base gauss --num_epochs 30 --loss mime --learning_rate 0.0001