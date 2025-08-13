import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, ReLU, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

def unet_2d(input_shape, num_filters, num_classes, batchnorm=False):
    
    initializer = HeNormal(42)

    inp = Input(shape=input_shape, dtype=tf.float64)
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(inp)
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_1 = x

    x = Conv2D(num_filters * 2, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_2 = x

    x = Conv2D(num_filters * 4, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_3 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_4 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x_5 = x

    x = Conv2D(num_filters * 8, strides=(2, 2), kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    if (batchnorm):
        x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_5])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 8, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_4])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_3])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters * 2, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_2])
    if (batchnorm):
        x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # x = Conv2DTranspose(x.shape[-1], (3,3), strides=(2,2), padding='same')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_1])
    x = Conv2D(num_classes, kernel_size=1, padding='same', kernel_initializer=initializer, activation=None)(x)
    out = Activation('softmax')(x)

    return Model(inp, out)