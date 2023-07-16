#!/bin/sh
python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.000001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.000001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.000001

python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.000005
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.000005
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.000005

python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.00001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.00001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.00001

python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.00005
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.00005
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.00005

python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.0001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.0001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.0001

python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.0005
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.0005
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.0005

python training.py --gpu 0 --base gauss --num_epochs 15 --loss dice --learning_rate 0.001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss cross_entropy --learning_rate 0.001
python training.py --gpu 0 --base gauss --num_epochs 15 --loss mime --learning_rate 0.001