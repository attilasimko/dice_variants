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
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.clip_by_value(y_true, tf.constant(1e-12, y_pred.dtype), tf.constant(1.0, y_pred.dtype))
        ce_map = ce(y_true, y_pred)
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


def coin_loss(alphas, betas, epsilon=1.0):
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
        U  = Sy + Sp + eps                          # (B,C)

        B, C = tf.shape(I)[0], tf.shape(I)[1]
        dtype = y_pred.dtype

        # ---- build alpha per class ----
        # alphas can be scalar, list/tuple/array of str/float, or None/"-"
        if alphas is None:
            alpha_auto = coin_a(U)                              # (B,C)
            alpha = tf.stop_gradient(alpha_auto)
        else:
            # turn ["-", 0.5, "-"] -> tensor with NaNs where auto, numbers where fixed
            if np.isscalar(alphas):
                alpha_fixed_vec = tf.fill([C], tf.cast(float(alphas), dtype))
                mask_fixed = tf.ones([C], dtype=tf.bool)
            else:
                av = []
                mv = []
                for a in list(alphas):
                    if a is None or (isinstance(a, str) and a.strip() == "-"):
                        av.append(np.nan); mv.append(False)
                    else:
                        av.append(float(a)); mv.append(True)
                alpha_fixed_vec = tf.constant(av, dtype=dtype)         # (C,)
                mask_fixed = tf.constant(mv, dtype=tf.bool)            # (C,)
            # broadcast to (B,C)
            alpha_fixed = tf.broadcast_to(alpha_fixed_vec, tf.shape(U))
            alpha_auto  = coin_a(U)
            # choose fixed where provided, else auto
            alpha = tf.where(tf.broadcast_to(mask_fixed, tf.shape(U)),
                             alpha_fixed, alpha_auto)
            alpha = tf.stop_gradient(alpha)

        # ---- build beta per class ----
        if betas is None:
            beta_auto = coin_b(U, I)
            beta = tf.stop_gradient(beta_auto)
        else:
            if np.isscalar(betas):
                beta_fixed_vec = tf.fill([C], tf.cast(float(betas), dtype))
                mask_fixed_b = tf.ones([C], dtype=tf.bool)
            else:
                bv = []
                mvb = []
                for b in list(betas):
                    if b is None or (isinstance(b, str) and b.strip() == "-"):
                        bv.append(np.nan); mvb.append(False)
                    else:
                        bv.append(float(b)); mvb.append(True)
                beta_fixed_vec = tf.constant(bv, dtype=dtype)          # (C,)
                mask_fixed_b = tf.constant(mvb, dtype=tf.bool)         # (C,)
            beta_fixed = tf.broadcast_to(beta_fixed_vec, tf.shape(U))
            beta_auto  = coin_b(U, I)
            beta = tf.where(tf.broadcast_to(mask_fixed_b, tf.shape(U)),
                            beta_fixed, beta_auto)
            beta = tf.stop_gradient(beta)

        # scalar per class whose grad wrt y_pred is -alpha*y + beta
        per_class = (-alpha * I) + (beta * Sp)              # (B,C)

        # present-class average (same reduction as typical per-class Dice)
        present = tf.cast(Sy > 0, dtype)
        denom = tf.maximum(K.sum(present, axis=-1, keepdims=True), 1.0)
        w = present / denom                                 # (B,C)

        per_sample = K.sum(per_class * w, axis=-1)          # (B,)
        return K.mean(per_sample)                           # scalar
    return loss_fn

# def coin_loss(_alphas, _betas, epsilon):
#     import tensorflow as tf
#     replace_alphas = []
#     alphas = []
#     betas = []
#     replace_betas = []
#     for _ in range(len(_alphas)):
#         replace_alphas.append(False)
#         replace_betas.append(False)
    
#     for i in range(len(_alphas)):
#         alphas.append(_alphas[i])
#         if (_alphas[i] == "-"):
#             replace_alphas[i] = True

#     for i in range(len(_betas)):
#         betas.append(_betas[i])
#         if (_betas[i] == "-"):
#             replace_betas[i] = True

#     def loss_fn(y_true, y_pred):
#         loss = 0.0
#         for slc in range(y_true.shape[0]):
#             for i in range(y_true.shape[3]):
#                 flat_true = tf.stop_gradient(K.flatten(y_true[slc, :, :, i]))
#                 flat_pred = tf.stop_gradient(K.flatten(y_pred[slc, :, :, i]))
#                 U = K.sum(flat_true) + K.sum(flat_pred) + epsilon
#                 I = K.sum(flat_true * flat_pred)

#                 if (replace_alphas[i]):
#                     alpha = tf.stop_gradient(tf.cast(2 / U, tf.float64))
#                 else:
#                     alpha = float(alphas[i])
#                     # Best so far: alpha = tf.stop_gradient(tf.cast(2 / (K.sum(flat_pred) + epsilon), tf.float64))

#                 if (replace_betas[i]):
#                     beta = tf.stop_gradient(tf.cast(2 * I / (U * U), tf.float64))
#                 else:
#                     beta = float(betas[i])
#                     # Best so far: beta = tf.stop_gradient(tf.cast(2 / (val_mean * val_mean), tf.float64))

            
#                 loss += K.sum((- alpha * y_true[slc, :, :, i] * y_pred[slc, :, :, i]) + (beta * y_pred[slc, :, :, i]))

#         return loss / y_true.shape[0]
#     return loss_fn
