from keras import backend as K
import tensorflow as tf
import numpy as np

def mime_loss(y_true, y_pred):
    mask = tf.greater(y_true, 0.0)
    loss = - y_true * y_pred
    loss_sum = tf.reduce_sum(loss[mask])
    if (tf.reduce_sum(tf.cast(mask, tf.float32)) > 0.0):
        loss_sum = loss_sum / tf.cast(tf.size(loss[mask]), tf.float32)
    return loss_sum


def compile(model, optimizer_str, lr_str, loss_str):
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
        loss = mime_loss
    
    model.compile(loss=loss, metrics=[dice_coef, dice_coef_a, dice_coef_b], optimizer=optimizer)

def dice_coef_a(y_true, y_pred, smooth=0.001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection

def dice_coef_b(y_true, y_pred, smooth=0.001):
    y_true_f = K.flatten(y_true)
    dice = smooth / (K.sum(y_pred) + smooth)
    return dice

def dice_coef(y_true, y_pred, smooth=0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return dice

def dice_loss(y_true, y_pred, smooth=0.001):
    return 1 - dice_coef(y_true, y_pred, smooth)

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

def dice_squared_loss(y_true, y_pred, smooth=0.1):    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.square(y_true_f) + K.square(y_pred_f))
    difference = K.sum(K.square(y_true_f - y_pred_f))
    dice = ((difference / intersection) + difference / (tf.cast(tf.size(y_true_f), tf.float32)))
    return dice

def cross_entropy_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    cross_entropy = -K.sum(y_true_f * K.log(y_pred_f))
    return cross_entropy