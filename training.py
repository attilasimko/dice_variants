from comet_ml import Experiment
import argparse
import os
from keras import backend as K

# Set up argument parser for running code from terminal
parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--dataset", default="WMH", help="Select dataset. Options are 'acdc' and 'wmh'.")
parser.add_argument("--num_epochs", default=10, help="Number of epochs.")
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate for the optimizer used during training. (Adam, SGD, RMSprop)")
parser.add_argument("--loss", default="mime", help="Loss function to use during training.")
parser.add_argument("--alpha1", default=0, help="Alpha for mime loss.")
parser.add_argument("--beta1", default=1, help="Beta for mime loss.")
parser.add_argument("--alpha2", default=0, help="Alpha for mime loss.")
parser.add_argument("--beta2", default=1, help="Beta for mime loss.")
parser.add_argument("--alpha3", default=0, help="Alpha for mime loss.")
parser.add_argument("--beta3", default=1, help="Beta for mime loss.")
parser.add_argument("--mimick", default="False", help="Beta for mime loss.")
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

# Set seeds for reproducibility
set_seeds()
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

# Allow gpu memory growth for tracking
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Log training parameters to the experiment
experiment.log_parameter("dataset", dataset) # The dataset used (MIQA or MIQAtoy)
experiment.log_parameter("loss", args.loss) # The loss function used
experiment.log_parameter("alpha1", float(args.alpha1)) # Alpha for mime loss
experiment.log_parameter("beta1", float(args.beta1)) # Beta for mime loss
experiment.log_parameter("alpha2", float(args.alpha2)) # Alpha for mime loss
experiment.log_parameter("beta2", float(args.beta2)) # Beta for mime loss
experiment.log_parameter("alpha3", float(args.alpha3)) # Alpha for mime loss
experiment.log_parameter("beta3", float(args.beta3)) # Beta for mime loss
experiment.log_parameter("mimick", args.mimick) # Beta for mime loss
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
compile(model, experiment.get_parameter('optimizer'), 
        experiment.get_parameter('learning_rate'), 
        experiment.get_parameter('loss'), 
        float(experiment.get_parameter('alpha1')), 
        float(experiment.get_parameter('alpha2')), 
        float(experiment.get_parameter('alpha3')), 
        float(experiment.get_parameter('beta1')), 
        float(experiment.get_parameter('beta2')), 
        float(experiment.get_parameter('beta3')), 
        batch_size * 256 * 256 * 2,
        experiment.get_parameter('mimick') == "True")

print("Trainable model weights:")
print(int(np.sum([K.count_params(p) for p in model.trainable_weights])))

x_val, y_val = gen_val.get_patient_data()
for epoch in range(num_epochs):
    experiment.set_epoch(epoch)
    loss_total = []
    loss_a = []
    loss_b = []

    for i in range(int(len(gen_train))):
        x, y = gen_train.next_batch()
        loss, loss_alpha, loss_beta = model.train_on_batch(x, y)
        
        loss_total.append(loss)
        loss_a.append(loss_alpha)
        loss_b.append(loss_beta)

    gen_train.stop()
    experiment.log_metrics({'training_loss': np.mean(loss_total),
                            'training_dice_a': np.mean(loss_a),
                            'training_dice_b': np.mean(loss_b)}, epoch=epoch)
    print(f"Training - Loss: {str(np.mean(np.mean(loss_total)))}")
    evaluate(experiment, (x_val, y_val), model, "val", labels, epoch)
    
    gen_val.stop()
    K.clear_session()
    gc.collect()

x_test, y_test = gen_test.get_patient_data()
evaluate(experiment, (x_test, y_test), model, "test", labels, epoch)
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