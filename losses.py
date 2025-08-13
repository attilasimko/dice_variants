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
    return (K.sum(y**2) + K.sum(s**2)) + epsilon

def coin_I(y, s):
    return K.sum(y * s)

# def coin_loss(epsilon=1.0):
#     def loss_fn(y_true, y_pred):
#         loss = 0.0
#         for slc in range(y_true.shape[0]):
#             for i in range(y_true.shape[3]):
#                 flat_true = tf.stop_gradient(K.flatten(y_true[slc, :, :, i]))
#                 flat_pred = tf.stop_gradient(K.flatten(y_pred[slc, :, :, i]))
#                 U = K.sum(flat_true) + K.sum(flat_pred) + epsilon
#                 I = K.sum(flat_true * flat_pred)
#                 alpha = tf.stop_gradient(tf.cast(2 / U, tf.float64))
#                 beta = tf.stop_gradient(tf.cast(2 * I / (U * U), tf.float64))
#                 loss += K.sum((- alpha * y_true[slc, :, :, i] * y_pred[slc, :, :, i]) + (beta * y_pred[slc, :, :, i]))

#         return loss
#     return loss_fn

def coin_loss(epsilon=1.0):
    """
    Multi-class Coin loss with per-class alpha/beta:
      - Pass a list/tuple/np.array of length C with either float values or "-"
      - Use "-" (or None) to auto-compute alpha=2/U and beta=2I/U^2 per sample/class
      - Provided numbers are treated as constants (no grads)
    Shapes: y_true, y_pred: (B, H, W, C)
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        eps = tf.cast(epsilon, y_pred.dtype)

        # per-sample, per-class sums
        I  = K.sum(y_true * y_pred, axis=[1, 2])   # (B,C)
        Sy = K.sum(y_true,           axis=[1, 2])  # (B,C)
        Sp = K.sum(y_pred,           axis=[1, 2])  # (B,C)
        U  = Sy + Sp + eps                           # (B,C)

        # auto alpha/beta (no grads)
        U_sg = tf.stop_gradient(U)
        I_sg = tf.stop_gradient(I)

        alpha = coin_a(U_sg)
        beta  = coin_b(U_sg, I_sg)

        # scalar per class whose grad wrt y_pred is -alpha*y + beta
        per_class = (-alpha * I) + (beta * Sp)      # (B,C)
        # present-class average (same reduction as typical per-class Dice)
        present = tf.cast(Sy > 0, y_pred.dtype)
        denom = tf.maximum(K.sum(present, axis=-1, keepdims=True), 1.0)
        w = present / denom                         # (B,C)

        per_sample = K.sum(per_class * w, axis=-1)  # (B,)
        return K.mean(per_sample)
    return loss_fn

def dice_coef(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.expand_dims(coin_I(y_true_f, y_pred_f), 0)
    union = tf.expand_dims(coin_U(y_true_f, y_pred_f, epsilon), 0)
    return 2. * intersection / union

# def dice_loss(epsilon=1.0):
#     def loss_fn(y_true, y_pred):
#         loss = 0.0
#         for slc in range(y_true.shape[0]):
#             for i in range(y_true.shape[3]):
#                 loss += 1 - dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
#         return loss
#     return loss_fn

def dice_loss(epsilon=1):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        eps = tf.cast(epsilon, y_pred.dtype)

        # per-sample, per-class stats
        I  = K.sum(y_true * y_pred, axis=[1,2])      # (B,C)
        Sy = K.sum(y_true,           axis=[1,2])     # (B,C)
        Sp = K.sum(y_pred,           axis=[1,2])     # (B,C)
        U  = Sy + Sp + eps                            # (B,C)

        dice_pc = 2.0 * I / U                         # (B,C)
        per_class_loss = 1.0 - dice_pc

        # average over classes that are present in each sample
        present = tf.cast(Sy > 0, y_pred.dtype)       # (B,C)
        denom = tf.maximum(K.sum(present, axis=-1, keepdims=True), 1.0)
        w = present / denom                           # (B,C)

        per_sample = K.sum(per_class_loss * w, axis=-1)  # (B,)
        return K.mean(per_sample)
    return loss_fn

def squared_dice_coef(y_true, y_pred, epsilon=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.expand_dims(coin_I(y_true_f, y_pred_f), 0)
    union = tf.expand_dims(coin_U_squared(y_true_f, y_pred_f, epsilon), 0)
    return 2. * intersection / union

def squared_dice_loss(epsilon=1):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        eps = tf.cast(epsilon, y_pred.dtype)

        I   = K.sum(y_true * y_pred,  axis=[1,2])        # (B,C)
        Sy2 = K.sum(K.square(y_true), axis=[1,2])        # (B,C)
        Sp2 = K.sum(K.square(y_pred), axis=[1,2])        # (B,C)
        U   = Sy2 + Sp2 + eps                             # (B,C)

        dice_pc = 2.0 * I / U
        per_class_loss = 1.0 - dice_pc

        present = tf.cast(K.sum(y_true, axis=[1,2]) > 0, y_pred.dtype)
        denom = tf.maximum(K.sum(present, axis=-1, keepdims=True), 1.0)
        w = present / denom

        per_sample = K.sum(per_class_loss * w, axis=-1)
        return K.mean(per_sample)
    return loss_fn

def cross_entropy_loss():
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')

    def loss_fn(y_true, y_pred):
        # Match dtype
        y_pred = tf.cast(y_pred, tf.float64 if y_pred.dtype == tf.float64 else tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)

        # 1) Scrub non-finite probs *before* any log is taken
        p = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)

        # 2) Re-normalize per pixel (so channels sum to 1). If sum==0, leave as zeros for now.
        sum_p = tf.reduce_sum(p, axis=-1, keepdims=True)
        nonzero = sum_p > 0
        p = tf.where(nonzero, p / sum_p, p)

        # 3) Clip away exact zeros/ones to avoid log(0); epsilon depends on dtype
        eps = tf.constant(1e-12 if p.dtype == tf.float64 else 1e-7, p.dtype)
        p = tf.clip_by_value(p, eps, 1.0)

        # 4) Optional: mask out invalid labels (not exactly one-hot)
        valid = tf.reduce_sum(y_true, axis=-1)                          # (B,H,W)
        valid = tf.cast(tf.where(tf.abs(valid - 1.0) < 1e-6, 1.0, 0.0), tf.float64)          # 1 if one-hot, else 0

        # 5) Call the built-in CE (returns per-pixel loss map)
        ce_map = ce(y_true, p)                                          # (B,H,W)
        ce_map = tf.where(tf.math.is_finite(ce_map), ce_map, 0.0)       # belt & suspenders

        # 6) Masked mean (avoid 0/0)
        denom = tf.reduce_sum(valid) + 1e-8
        return tf.reduce_sum(ce_map * valid) / denom
    return loss_fn

def dice_ce_loss(epsilon=1):
    ce_loss = cross_entropy_loss()
    dice_loss_fn = dice_loss(epsilon)
    def loss_fn(y_true, y_pred):
        return ce_loss(y_true, y_pred) + dice_loss_fn(y_true, y_pred)
    return loss_fn

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

