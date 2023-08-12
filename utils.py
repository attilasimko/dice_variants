from keras import backend as K
import tensorflow as tf
import numpy as np

def compile(model, optimizer_str, lr_str, loss_str, alpha=1, beta=1):
    import tensorflow

    lr = float(lr_str)
    if optimizer_str == 'Adam':
        optimizer = tensorflow.keras.optimizers.Adam(lr)
    elif optimizer_str == 'SGD':
        optimizer = tensorflow.keras.optimizers.SGD(lr, momentum=0.99, weight_decay=3e-5)
    elif optimizer_str == 'RMSprop':
        optimizer = tensorflow.keras.optimizers.RMSprop(lr)
    else:
        raise NotImplementedError
    
    if loss_str == 'dice':
        loss = dice_loss
    elif loss_str == 'cross_entropy':
        loss = tf.keras.losses.CategoricalCrossentropy()
    elif loss_str == "mime":
        mime = mime_loss(alpha, beta)
        loss = mime
    
    model.compile(loss=loss, metrics=[dice_coef, dice_coef_a, dice_coef_b], optimizer=optimizer)

def dice_coef_a(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((intersection + smooth) / (K.sum(y_true_f) + smooth))

def dice_coef_b(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return ((K.sum(y_true_f) + K.sum(y_pred_f) + smooth) / (K.sum(y_true_f) + smooth))

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(K.cast(y_true, tf.float32))
    y_pred_f = K.flatten(K.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return dice

def dice_loss(y_true, y_pred, smooth=100):
    loss = 0.0
    num_el = 0.0
    for slc in range(np.shape(y_true)[0]):
        for i in range(np.shape(y_true)[3]):
                loss += 1 - dice_coef(y_true[slc:slc+1, :, :, i], y_pred[slc:slc+1, :, :, i], smooth)
                num_el += 1
    return loss / num_el

def mime_loss(a=1, b=1):
    def loss_fn(y_true, y_pred):
        loss = 0.0
        num_el = 0.0
        for slc in range(np.shape(y_true)[0]):
            for i in range(np.shape(y_true)[3]):
                mask_a = tf.not_equal(y_true[slc, :, :, i], 0.0)
                mask_b = tf.equal(y_true[slc, :, :, i], 0.0)
                loss_a = - a * y_pred[slc, :, :, i][mask_a]
                loss_b = b * y_pred[slc, :, :, i][mask_b]
                if (~tf.math.is_nan(tf.reduce_mean(loss_a))):
                    loss += tf.reduce_mean(loss_a)
                    num_el += 1
                if (~tf.math.is_nan(tf.reduce_mean(loss_b))):
                    loss += tf.reduce_mean(loss_b)
                    num_el += 1
        return loss / num_el
    return loss_fn

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

def dice_squared_loss(y_true, y_pred, smooth=0.1):    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.square(y_true_f) + K.square(y_pred_f))
    difference = K.sum(K.square(y_true_f - y_pred_f))
    dice = ((difference / intersection) + difference / (tf.cast(tf.size(y_true_f), tf.float32)))
    return dice