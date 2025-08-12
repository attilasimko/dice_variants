from sympy import N
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np

def _maybe_softmax(y_pred, from_logits):
    return tf.nn.softmax(y_pred, axis=-1) if from_logits else y_pred


def _weights_present_classes(y_true):
    # (B,H,W,C) one-hot -> weight 1/num_present for present classes, else 0
    present = tf.reduce_sum(y_true, axis=[1, 2]) > 0            # (B,C) bool
    present = tf.cast(present, y_true.dtype)
    denom = tf.reduce_sum(present, axis=-1, keepdims=True)      # (B,1)
    denom = tf.maximum(denom, 1.0)
    return present / denom                                      # (B,C)

def get_coeffs(y_true, y_pred):
    return coin_I(y_true, y_pred).numpy(), coin_U(y_true, y_pred).numpy(), coin_coef_a(y_true, y_pred).numpy(), coin_coef_b(y_true, y_pred).numpy()
    
def coin_coef_a(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    U = coin_U(y_true_f, y_pred_f, epsilon)
    return coin_a(U)


def coin_coef_b(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    I = coin_I(y_true_f, y_pred_f)
    U = coin_U(y_true_f, y_pred_f, epsilon)
    return coin_b(U, I)

def coin_a(u):
    return 2 / u

def coin_b(u, i):
    return 2 * i / (u * u)

def coin_U(y, s, epsilon=1):
    return (K.sum(y) + K.sum(s)) + epsilon

def coin_U_squared(y, s, epsilon=1):
    return (K.sum(y**2) + K.sum(s**2)) + epsilon

def coin_I(y, s):
    return K.sum(y * s)

def dice_coef(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.expand_dims(coin_I(y_true_f, y_pred_f), 0)
    union = tf.expand_dims(coin_U(y_true_f, y_pred_f, epsilon), 0)
    return 2. * intersection / union

def dice_loss(epsilon=1):
    def loss_fn(y_true, y_pred):
        loss = 0.0
        for slc in range(y_true.shape[0]):
            for i in range(y_true.shape[3]):
                loss += 1 - dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
        return loss / y_true.shape[0]
    return loss_fn

def squared_dice_coef(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.expand_dims(coin_I(y_true_f, y_pred_f), 0)
    union = tf.expand_dims(coin_U_squared(y_true_f, y_pred_f, epsilon), 0)
    return 2. * intersection / union

def squared_dice_loss(epsilon=1):
    def loss_fn(y_true, y_pred):
        loss = 0.0
        iter = 0
        for slc in range(y_true.shape[0]):
            for i in range(y_true.shape[3]):
                loss += 1 - squared_dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
                iter += 1
        return loss / iter
    return loss_fn

def cross_entropy_loss():
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        ce_map = ce(y_true, y_pred)              # (B,H,W)
        return K.mean(ce_map)
    return loss_fn

def dice_ce_loss(epsilon=1):
    ce_loss = cross_entropy_loss()
    dice_loss_fn = dice_loss(epsilon)
    def loss_fn(y_true, y_pred):
        return ce_loss(y_true, y_pred) + dice_loss_fn(y_true, y_pred)
    return loss_fn

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

def coin_loss(_alphas, _betas, epsilon):
    import tensorflow as tf
    replace_alphas = []
    alphas = []
    betas = []
    replace_betas = []
    for _ in range(len(_alphas)):
        replace_alphas.append(False)
        replace_betas.append(False)
    
    for i in range(len(_alphas)):
        alphas.append(_alphas[i])
        if (_alphas[i] == "-"):
            replace_alphas[i] = True

    for i in range(len(_betas)):
        betas.append(_betas[i])
        if (_betas[i] == "-"):
            replace_betas[i] = True

    def loss_fn(y_true, y_pred):
        
        loss = 0.0
        for slc in range(y_true.shape[0]):
            for i in range(y_true.shape[3]):
                flat_true = tf.stop_gradient(K.flatten(y_true[slc, :, :, i]))
                flat_pred = tf.stop_gradient(K.flatten(y_pred[slc, :, :, i]))
                U = K.sum(flat_true) + K.sum(flat_pred) + epsilon
                I = K.sum(flat_true * flat_pred)

                if (replace_alphas[i]):
                    alpha = tf.stop_gradient(tf.cast(2 / U, tf.float64))
                else:
                    alpha = float(alphas[i])
                    # Best so far: alpha = tf.stop_gradient(tf.cast(2 / (K.sum(flat_pred) + epsilon), tf.float64))

                if (replace_betas[i]):
                    beta = tf.stop_gradient(tf.cast(2 * I / (U * U), tf.float64))
                else:
                    beta = float(betas[i])
                    # Best so far: beta = tf.stop_gradient(tf.cast(2 / (val_mean * val_mean), tf.float64))

            
                loss += K.sum((- alpha * y_true[slc, :, :, i] * y_pred[slc, :, :, i]) + (beta * y_pred[slc, :, :, i]))

        return loss / y_true.shape[0]
    return loss_fn
