from sympy import N
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np

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
    return (K.sum(y)**2 + K.sum(s)**2) + epsilon

def coin_I(y, s):
    return K.sum(y * s)

def dice_coef(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.expand_dims(coin_I(y_true_f, y_pred_f), 0)
    union = tf.expand_dims(coin_U(y_true_f, y_pred_f, epsilon), 0)
    return 2. * intersection / union

def squared_dice_coef(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.expand_dims(coin_I(y_true_f, y_pred_f), 0)
    union = tf.expand_dims(coin_U_squared(y_true_f, y_pred_f, epsilon), 0)
    return 2. * intersection / union

def cross_entropy_loss(skip_background=False):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        iter = 0
        for slc in range(y_true.shape[0]):
            loss += tf.losses.categorical_crossentropy(y_true[slc, :, :, start_idx:], y_pred[slc, :, :, start_idx:])
            iter += 1
        return loss / y_true.shape[0]
    return loss_fn

def dice_loss(skip_background=False, epsilon=1):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        for slc in range(y_true.shape[0]):
            for i in range(start_idx, y_true.shape[3]):
                loss += 1 - dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
        return loss / y_true.shape[0]
    return loss_fn

def dice_ce_loss(skip_background=False, epsilon=1):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        for slc in range(y_true.shape[0]):
            loss += tf.losses.categorical_crossentropy(y_true[slc, :, :, start_idx:], y_pred[slc, :, :, start_idx:])
            for i in range(start_idx, y_true.shape[3]):
                loss += 1 - dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
        return loss / y_true.shape[0]
    return loss_fn

def squared_dice_loss(skip_background=False, epsilon=1):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        iter = 0
        for slc in range(y_true.shape[0]):
            for i in range(start_idx, y_true.shape[3]):
                loss += 1 - squared_dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
                iter += 1
        return loss / iter
    return loss_fn

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
        # avg_sums = np.multiply([0.9684, 0.0106, 0.0102, 0.0108], 65536.0) # ["Background", "LV", "RV", "Myo"]
        avg_sums = np.multiply([0.9694, 0.0102, 0.0102, 0.0102], 65536.0) # ["Background", "LV", "RV", "Myo"]
        avg_as = [-1.5756e-5, -0.00144, -0.00150, -0.00141]
        avg_bs = [65536.0, 1.0, 1.0, 1.0]
        # avg_sums = np.multiply([0.9997, 0.0001, 0.0001, 0.0001], 65536.0)
        if (np.sum(avg_sums) != 65536.0):
            raise ValueError("Sum of averages is not 1.0")
        
        loss = 0.0
        for slc in range(y_true.shape[0]):
            # worst_dice = 1.0
            # worst_idx = 0
            # for i in range(y_true.shape[3]):
            #     if (K.sum(y_true[slc, :, :, i]) > 0):
            #         current_dice = dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i])
            #         if (current_dice < worst_dice):
            #             worst_dice = current_dice
            #             worst_idx = i

            # x_idx = K.random.randint(0, y_true.shape[1]-1)
            # y_idx = K.random.randint(0, y_true.shape[2]-1)

            for i in range(y_true.shape[3]):
                flat_true = tf.stop_gradient(K.flatten(y_true[slc, :, :, i]))
                flat_pred = tf.stop_gradient(K.flatten(y_pred[slc, :, :, i]))
                U = K.sum(flat_true) + K.sum(flat_pred) + epsilon
                I = K.sum(flat_true * flat_pred)
                mask = tf.cast(tf.less(y_pred[slc, :, :, i], 0.75), tf.float64)
                val_mean = avg_sums[i]

                if (replace_alphas[i]):
                    alpha = tf.stop_gradient(tf.cast(2 / U, tf.float64))
                    # Best so far: alpha = tf.stop_gradient(tf.cast(2 / (K.sum(flat_pred) + epsilon), tf.float64))
                else:
                    alpha = float(alphas[i])

                if (replace_betas[i]):
                    beta = tf.stop_gradient(tf.cast(2 * I / (U * U), tf.float64))
                    # Best so far: beta = tf.stop_gradient(tf.cast(2 / (val_mean * val_mean), tf.float64))
                else:
                    beta = float(betas[i])

                # if (tf.reduce_any(alpha < beta)):
                #     raise ValueError("Positive gradient overflow. Alpha < Beta")
            
                loss += K.sum((- alpha * y_true[slc, :, :, i] * y_pred[slc, :, :, i]) + (beta * y_pred[slc, :, :, i]))

        return loss / y_true.shape[0]
    return loss_fn

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

def dice_squared_loss(y_true, y_pred, epsilon=0.1):    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.square(y_true_f) + K.square(y_pred_f) + epsilon)
    difference = K.sum(K.square(y_true_f - y_pred_f))
    dice = ((difference / intersection) + difference / (tf.cast(tf.size(y_true_f), tf.float64)))
    return dice
