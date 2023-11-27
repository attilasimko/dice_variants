from comet_ml import Experiment
import argparse
import os
from keras import backend as K
import sys
sys.path.insert(1, os.path.abspath('.'))

# Set up argument parser for running code from terminal
parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--dataset", default="ACDC", help="Select dataset. Options are 'acdc' and 'wmh'.")
parser.add_argument("--num_epochs", default=10, help="Number of epochs.")
parser.add_argument("--optimizer", default="Adam", help="Optimizer to use during training.")
parser.add_argument("--batch_size", default=12, help="Batch size for training and validating.")
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate for the optimizer used during training. (Adam, SGD, RMSprop)")
parser.add_argument("--loss", default="coin", help="Loss function to use during training.")
parser.add_argument("--alpha1", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta1", default="-", help="Beta for coin loss.")
parser.add_argument("--alpha2", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta2", default="-", help="Beta for coin loss.")
parser.add_argument("--alpha3", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta3", default="-", help="Beta for coin loss.")
parser.add_argument("--alpha4", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta4", default="-", help="Beta for coin loss.")
parser.add_argument("--base", default=None) # Name of my PC, used to differentiate between different paths.
parser.add_argument("--gpu", default=None) # If using gauss, you need to specify the GPU to use.
args = parser.parse_args()

if ((args.dataset == "ACDC") | (args.dataset == "WMH")):
    dataset = args.dataset
else:
    raise ValueError("Dataset not found.")
gpu = args.gpu
if args.base == "gauss":
    if gpu is not None: # GPUs are usually defined only when running from gauss, everywhere else there's only a single GPU.
        base = 'gauss'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        base_path = '/data/attila/' + str(dataset) + '_0/'
        save_path = '/home/attila/out/'
    else:
        raise ValueError("You need to specify a GPU to use on gauss.")
elif args.base == "alvis":
    base = 'alvis'
    base_path = '/mimer/NOBACKUP/groups/naiss2023-6-64/' + str(dataset) + '_0/'
    save_path = '/cephyr/users/attilas/Alvis/out/'
else:
    base_path = "/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/" + dataset + '_0/'
    save_path = "/home/attilasimko/Documents/out/"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
from data import DataGenerator
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import uuid
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import gc
from keras import backend as K
from utils import *

# Set seeds for reproducibility
set_seeds()
batch_size = int(args.batch_size)
num_epochs = int(args.num_epochs)
if (dataset == "WMH"):
    labels = ["Background", "WMH", "Other"]
elif (dataset == "ACDC"):
    labels = ["Background", "LV", "RV", "Myo"]

# Data generator for validation data
gen_val = DataGenerator(base_path + "train/",
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False)

plot_idx = 0

metric_U = []
metric_I = []
metric_dice_a = []
metric_dice_b = []

for label in labels:
    metric_U.append([])
    metric_I.append([])
    metric_dice_a.append([])
    metric_dice_b.append([])

for i in range(int(len(gen_val))):
    x, y = gen_val.next_batch()
    for j in range(np.shape(y)[3]):
        current_y = y[:, :, :, j].astype(np.float64)
        for slc in range(np.shape(current_y)[0]):
            if (np.sum(current_y[slc, :, :]) == 1):
                print(gen_val.temp_ID[slc])
                print(coin_coef_a(K.flatten(current_y[slc, :, :]), K.flatten(current_y[slc, :, :])))
                print(coin_coef_b(K.flatten(current_y[slc, :, :]), K.flatten(current_y[slc, :, :])))
            metric_U[j].append(coin_U(K.flatten(current_y[slc, :, :]), K.flatten(current_y[slc, :, :]), 0).numpy())
            metric_I[j].append(coin_I(K.flatten(current_y[slc, :, :]), K.flatten(current_y[slc, :, :])).numpy())
    
for j in range(len(labels)):
    print("Minimum:")
    print(labels[j])
    print(str(np.min(np.array(metric_U[j]))))
    print(str(np.min(np.array(metric_I[j]))))


for j in range(len(labels)):
    print("Maximum:")
    print(labels[j])
    print(str(np.max(np.array(metric_U[j]))))
    print(str(np.max(np.array(metric_I[j]))))


for j in range(len(labels)):
    print("Mean:")
    print(labels[j])
    print(str(np.mean(np.array(metric_U[j]))) + "+-" + str(np.std(np.array(metric_U[j]))))
    print(str(np.mean(np.array(metric_I[j]))) + "+-" + str(np.std(np.array(metric_I[j]))))
    print("a: " + str(coin_a(np.mean(np.array(metric_U[j])))))
    print("b: " + str(coin_b(np.mean(np.array(metric_U[j])), np.mean(np.array(metric_I[j])))))
