from comet_ml import Experiment
import argparse
import os

# Set up argument parser for running code from terminal
parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--dataset", default="WMH", help="Select dataset. Options are 'acdc' and 'wmh'.")
parser.add_argument("--num_epochs", default=10, help="Number of epochs.")
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate for the optimizer used during training. (Adam, SGD, RMSprop)")
parser.add_argument("--loss", default="dice", help="Loss function to use during training.")
parser.add_argument("--alpha", default=1, help="Alpha for mime loss.")
parser.add_argument("--beta", default=1, help="Beta for mime loss.")
parser.add_argument("--optimizer", default="Adam", help="Optimizer to use during training.")
parser.add_argument("--batch_size", default=12, help="Batch size for training and validating.")
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
from model import unet_2d
from data import DataGenerator
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import uuid
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import gc
from keras import backend as K
from utils import *

# All the comet_ml things are for online progress tracking, with this API key you get access to the MIQA project
experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="dice_variants")
batch_size = args.batch_size
num_epochs = int(args.num_epochs)
labels = ["Background", "WMH", "Other"]
# Custom data generator for efficiently loading the training data (stored as .npz files under base_path+"training/")
gen_train = DataGenerator(base_path + "train/",
                          batch_size=batch_size,
                          shuffle=True)

# Data generator for validation data
gen_val = DataGenerator(base_path + "val/",
                        batch_size=batch_size,
                        shuffle=False)

# Data generator for test data
gen_test = DataGenerator(base_path + "test/",
                         batch_size=1,
                         shuffle=False)


# Log training parameters to the experiment
experiment.log_parameter("dataset", dataset) # The dataset used (MIQA or MIQAtoy)
experiment.log_parameter("loss", args.loss) # The loss function used
experiment.log_parameter("alpha", float(args.alpha)) # Alpha for mime loss
experiment.log_parameter("beta", float(args.beta)) # Beta for mime loss
experiment.log_parameter("num_epochs", num_epochs) # The number of epochs
experiment.log_parameter("optimizer", args.optimizer)
experiment.log_parameter("learning_rate", float(args.learning_rate))
experiment.log_parameter("num_filters", int(12))

# Set up the generators. This could have been done before, but this way the generator is the same as in another project, which is conventient.
gen_train.set_experiment(experiment)
gen_val.set_experiment(experiment)
gen_test.set_experiment(experiment)

experiment_name = experiment.get_name()
if (experiment_name is None): #In some cases, comet_ml fails to provide a name for the experiment, in this case we generate a random UID
    experiment_name = str(uuid.uuid1())
print("Experiment started with name: " + str(experiment_name))

# Build model
model = unet_2d((256, 256, 2), 48, len(gen_train.outputs))
compile(model, experiment.get_parameter('optimizer'), experiment.get_parameter('learning_rate'), experiment.get_parameter('loss'), float(experiment.get_parameter('alpha')), float(experiment.get_parameter('beta')))

print("Trainable model weights:")
print(int(np.sum([K.count_params(p) for p in model.trainable_weights])))

for epoch in range(num_epochs):
    experiment.set_epoch(epoch)
    metric_dice = []
    metric_dice_a = []
    metric_dice_b = []
    loss_cnn = []

    for i in range(int(len(gen_train))):
        x, y = gen_train.next_batch()
        loss, metric, metric_a, metric_b = model.train_on_batch(x, y)
        loss_cnn.append(loss)
        metric_dice.append(100 * metric)
        metric_dice_a.append(100 * metric_a)
        metric_dice_b.append(100 * metric_b)
        
    gen_train.stop()
    experiment.log_metrics({'training_loss': np.mean(loss_cnn),
                            'training_dice': np.mean(metric_dice),
                            'training_dice_a': np.mean(metric_dice_a),
                            'training_dice_b': np.mean(metric_dice_b),
                            'training_dice_std': np.std(metric_dice),
                            'training_dice_a_std': np.std(metric_dice_a),
                            'training_dice_b_std': np.std(metric_dice_b)}, epoch=epoch)
    print(f"Training - Loss: {str(np.mean(np.mean(loss_cnn)))}")
    
    metric_dice = []
    metric_dice_a = []
    metric_dice_b = []
    metric_tp = []
    metric_tn = []
    metric_fp = []
    metric_fn = []
    for label in labels:
        metric_dice.append([])
        metric_dice_a.append([])
        metric_dice_b.append([])
        metric_tp.append([])
        metric_tn.append([])
        metric_fn.append([])
        metric_fp.append([])

    for i in range(len(gen_val)):
        x, y = gen_val.next_batch()
        pred = model.predict_on_batch(x)

        for slc in range(np.shape(y)[0]):
            for j in range(np.shape(y)[3]):
                current_y = y[slc:slc+1, :, :, j].astype(np.float32)
                current_pred = pred[slc:slc+1, :, :, j].astype(np.float32)
                metric_dice[j].append(dice_coef(current_y, current_pred).numpy())
                metric_dice_a[j].append(dice_coef_a(current_y, current_pred).numpy())
                metric_dice_b[j].append(dice_coef_b(current_y, current_pred).numpy())
                metric_tp[j].append(np.sum(current_y == 1) * (current_pred >= 0.5))
                metric_tn[j].append(np.sum(current_y == 0) * (current_pred < 0.5))
                metric_fp[j].append(np.sum(current_y == 0) * (current_pred >= 0.5))
                metric_fn[j].append(np.sum(current_y == 1) * (current_pred < 0.5))

    for j in range(len(labels)):
        metric_dice[j] = np.array(metric_dice[j])
        metric_dice_a[j] = np.array(metric_dice_a[j])
        metric_dice_b[j] = np.array(metric_dice_b[j])
        print(f"Validating Dice {labels[j]}: {np.mean(np.mean(metric_dice[j]))}")
        experiment.log_metrics({f'val_dice_{labels[j]}': np.mean(metric_dice[j]),
                                f'val_dice_{labels[j]}_std': np.std(metric_dice[j]),
                                f'val_dice_a_{labels[j]}': np.mean(metric_dice_a[j]),
                                f'val_dice_a_{labels[j]}_std': np.std(metric_dice_a[j]),
                                f'val_dice_b_{labels[j]}': np.mean(metric_dice_b[j]),
                                f'val_dice_b_{labels[j]}_std': np.std(metric_dice_b[j]),
                                f'val_tp_{labels[j]}': np.mean(metric_tp[j]),
                                f'val_tn_{labels[j]}': np.mean(metric_tn[j]),
                                f'val_fp_{labels[j]}': np.mean(metric_fp[j]),
                                f'val_fn_{labels[j]}': np.mean(metric_fn[j]),
                                f'val_tp_{labels[j]}_std': np.std(metric_tp[j]),
                                f'val_tn_{labels[j]}_std': np.std(metric_tn[j]),
                                f'val_fp_{labels[j]}_std': np.std(metric_fp[j]),
                                f'val_fn_{labels[j]}_std': np.std(metric_fn[j])}, epoch=epoch)
    gen_val.stop()
    K.clear_session()
    gc.collect()

for idx in range(len(gen_val)):
    x, y = gen_val.next_batch()

    if (np.sum(y[0, :, :, 1:]) == 0):
        continue

    pred = model.predict_on_batch(x)

    plt.subplot(251)
    plt.imshow(x[0, :, :, 0], cmap="gray", interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(252)
    plt.imshow(x[0, :, :, 1], cmap="gray", interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(253)
    plt.imshow(y[0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(254)
    plt.imshow(y[0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(255)
    plt.imshow(y[0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(256)
    plt.imshow(x[0, :, :, 0], cmap="gray", interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(257)
    plt.imshow(x[0, :, :, 1], cmap="gray", interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(258)
    plt.imshow(pred[0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(259)
    plt.imshow(pred[0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 5, 10)
    plt.imshow(pred[0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(save_path + str(idx) + ".png")
    experiment.log_image(save_path + str(idx) + ".png", step=epoch, overwrite=True)
experiment.end()