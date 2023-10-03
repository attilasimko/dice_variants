import math
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout,\
Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add , Concatenate, add, LeakyReLU, ReLU
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


def unet_2d(input_shape, num_filters, num_classes, batchnorm=False):
    policy = tf.keras.mixed_precision.Policy("float64")
    tf.keras.mixed_precision.set_global_policy(policy)

    inp = Input(shape=input_shape, dtype=tf.float64)
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='HeNormal')(inp)
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_1 = x

    x = Conv2D(num_filters * 2, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_2 = x

    x = Conv2D(num_filters * 4, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_3 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_4 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_5 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    if (batchnorm):
        x = BatchNormalization()(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 8, kernel_size=2, padding='same', kernel_initializer='HeNormal')(x)
    x = Concatenate()([x, x_5])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 8, kernel_size=2, padding='same', kernel_initializer='HeNormal')(x)
    x = Concatenate()([x, x_4])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 4, kernel_size=2, padding='same', kernel_initializer='HeNormal')(x)
    x = Concatenate()([x, x_3])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 2, kernel_size=2, padding='same', kernel_initializer='HeNormal')(x)
    x = Concatenate()([x, x_2])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters, kernel_size=2, padding='same', kernel_initializer='HeNormal')(x)
    x = Concatenate()([x, x_1])
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='HeNormal')(x)
    out = Conv2D(num_classes, kernel_size=1, padding='same', kernel_initializer='HeNormal', activation="softmax")(x)

    return Model(inp, out)