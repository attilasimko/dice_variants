import math
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout,\
Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , Concatenate, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax
import math
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, UpSampling2D, Conv2DTranspose, concatenate


def encoding_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
    # Proper initialization prevents from the problem of exploding and vanishing gradients 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv = Conv2D(n_filters, 
                3,   # Kernel size   
                activation='relu',
                padding='same',
                kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                3,   # Kernel size
                activation='relu',
                padding='same',
                kernel_initializer='HeNormal')(conv)
    
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection

def decoding_block(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                n_filters,
                (3,3),    # Kernel size
                strides=(2,2),
                padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters, 
                3,     # Kernel size
                activation='relu',
                padding='same',
                kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                3,   # Kernel size
                activation='relu',
                padding='same',
                kernel_initializer='HeNormal')(conv)
    return conv

def unet_2d(input_shape, num_filters, num_classes):
    x_skip = []
    dropout_rate = 0.2
    num_levels = 4
    inp = Input(input_shape)
    
    for i in range(num_levels + 1):
        if (i == 0):
            x_skip.insert(0, encoding_block(inp, num_filters,dropout_prob=0, max_pooling=True))
        elif (i == int(np.log2(inp.shape[1] / 16))):
            x_skip.insert(0, encoding_block(x_skip[0][0], num_filters*(2**i), dropout_prob=dropout_rate, max_pooling=False)) 
        else:
            x_skip.insert(0, encoding_block(x_skip[0][0],num_filters*(2**i),dropout_prob=0, max_pooling=True))
    
    ublock = x_skip[0][0]
    for i in range(num_levels):
        ublock = decoding_block(ublock, x_skip[i+1][1], int(x_skip[i][0].shape[-1]))
    out = Conv2D(num_classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock)
    out = Conv2D(num_classes, 1, padding='same', kernel_initializer='he_normal')(out)
    
    output = Activation('softmax')(out)

    model = Model(inp, output)
    return model