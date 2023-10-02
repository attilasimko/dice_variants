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
parser.add_argument("--loss", default="mime", help="Loss function to use during training.")
parser.add_argument("--round_off", default="4", help="Gradient round-off.")
parser.add_argument("--alpha1", default="-", help="Alpha for mime loss.")
parser.add_argument("--beta1", default="-", help="Beta for mime loss.")
parser.add_argument("--alpha2", default="-", help="Alpha for mime loss.")
parser.add_argument("--beta2", default="-", help="Beta for mime loss.")
parser.add_argument("--alpha3", default="-", help="Alpha for mime loss.")
parser.add_argument("--beta3", default="-", help="Beta for mime loss.")
parser.add_argument("--alpha4", default="-", help="Alpha for mime loss.")
parser.add_argument("--beta4", default="-", help="Beta for mime loss.")
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
        base_path = '/data/attila/' + str(dataset) + '/'
        save_path = '/home/attila/out/'
    else:
        raise ValueError("You need to specify a GPU to use on gauss.")
elif args.base == "alvis":
    base = 'alvis'
    base_path = '/mimer/NOBACKUP/groups/naiss2023-6-64/' + str(dataset) + '/'
    save_path = '/cephyr/users/attilas/Alvis/out/'
else:
    base_path = "/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/" + dataset + "/"
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
round_off = int(args.round_off)
num_epochs = int(args.num_epochs)
if (dataset == "WMH"):
    labels = ["Background", "WMH", "Other"]
elif (dataset == "ACDC"):
    labels = ["Background", "LV", "RV", "Myo"]

# Custom data generator for efficiently loading the training data (stored as .npz files under base_path+"training/")
gen_train = DataGenerator(base_path + "train/",
                          dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)

# Data generator for validation data
gen_val = DataGenerator(base_path + "val/",
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False)

# Data generator for test data
gen_test = DataGenerator(base_path + "test/",
                         dataset=dataset,
                         batch_size=1,
                         shuffle=False)

plot_idx = 0
loss_total = []
loss_a = []
loss_b = []
grads_min = []
grads_max = []

metric_dice = []
metric_dice_a = []
metric_dice_b = []
metric_tp = []
metric_tn = []
metric_fp = []
metric_fn = []

for label in labels:
    metric_dice_a.append([])
    metric_dice_b.append([])

for i in range(int(len(gen_val))):
    x, y = gen_val.next_batch()
    for j in range(np.shape(y)[3]):
        current_y = y[:, :, :, j].astype(np.float32)
        for idx in range(np.shape(current_y)[2]):
            if (np.sum(current_y[:, :, idx]) > 0):
                metric_dice_a[j].append(dice_coef_a(current_y[:, :, idx], current_y[:, :, idx]).numpy())
                metric_dice_b[j].append(dice_coef_b(current_y[:, :, idx], current_y[:, :, idx]).numpy())
    
for j in range(len(labels)):
    print(labels[j])
    print(str(np.mean(np.array(metric_dice_a[j]))) + "+-" + str(np.std(np.array(metric_dice_a[j]))))
    print(str(np.mean(np.array(metric_dice_b[j]))) + "+-" + str(np.std(np.array(metric_dice_b[j]))))