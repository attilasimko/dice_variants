import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, ReLU, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

from tensorflow.keras.layers import Conv2D, Add, ReLU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import backend as K

def residual_block(x, filters, kernel_size=3, stride=1, initializer=None, res_scale=1.0):
    """
    Simple residual block (no BatchNorm, no extra params besides convs).

    main:  Conv(k, stride) -> ReLU -> Conv(k)
    skip:  identity, or 1x1 conv with 'stride' to match shape

    Args:
        x: input tensor
        filters: output channels
        kernel_size: main conv kernel (int or tuple)
        stride: 1 for same resolution; >1 to downsample within the block
        initializer: kernel initializer (defaults to HeNormal(42))
        res_scale: optional constant to scale the residual branch (e.g., 0.1 for very deep nets)

    Returns:
        Tensor with shape matched via identity/projection, followed by ReLU.
    """
    if initializer is None:
        initializer = HeNormal(42)

    # --- main branch ---
    y = Conv2D(filters, kernel_size, strides=stride, padding='same',
               use_bias=True, kernel_initializer=initializer)(x)
    y = ReLU()(y)
    y = Conv2D(filters, kernel_size, padding='same',
               use_bias=True, kernel_initializer=initializer)(y)

    if res_scale != 1.0:
        y = tf.keras.layers.Lambda(lambda t: t * tf.cast(res_scale, t.dtype))(y)

    # --- skip branch (projection if shape/channel mismatch) ---
    in_ch = K.int_shape(x)[-1]
    need_proj = (stride != 1) or (in_ch is None or in_ch != filters)
    if need_proj:
        skip = Conv2D(filters, 1, strides=stride, padding='same',
                      use_bias=True, kernel_initializer=initializer)(x)
    else:
        skip = x

    # --- merge + activation ---
    out = Add()([y, skip])
    out = ReLU()(out)
    return out

def unet_2d(input_shape, num_filters, num_classes, batchnorm=False):
    # keep your policy/initializer
    initializer = HeNormal(42)

    inp = Input(shape=input_shape, dtype=tf.float64)

    # encoder
    x = residual_block(inp, num_filters, initializer=initializer)      # level 0
    x_1 = x

    x = residual_block(x, num_filters * 2, stride=2, initializer=initializer)  # level 1
    x_2 = x

    x = residual_block(x, num_filters * 4, stride=2, initializer=initializer)  # level 2
    x_3 = x

    x = residual_block(x, num_filters * 8, stride=2, initializer=initializer)  # level 3
    x_4 = x

    x = residual_block(x, num_filters * 8, stride=2, initializer=initializer)  # level 4
    x_5 = x

    x = residual_block(x, num_filters * 8, stride=2, initializer=initializer)  # bottom
    x = residual_block(x, num_filters * 8, initializer=initializer)

    # decoder (keep your upsampling; add a light conv before concat, then a residual block)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(num_filters * 8, 3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_5])
    x = residual_block(x, num_filters * 8, initializer=initializer)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(num_filters * 8, 3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_4])
    x = residual_block(x, num_filters * 8, initializer=initializer)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(num_filters * 4, 3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_3])
    x = residual_block(x, num_filters * 4, initializer=initializer)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(num_filters * 2, 3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_2])
    x = residual_block(x, num_filters * 2, initializer=initializer)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(num_filters, 3, padding='same', kernel_initializer=initializer)(x)
    x = Concatenate()([x, x_1])

    # head
    x = Conv2D(num_classes, 1, padding='same', kernel_initializer=initializer, activation=None)(x)
    out = Activation('softmax')(x)
    return Model(inp, out)
