from keras import backend as K
import tensorflow as tf
import numpy as np
import os

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42)

def compile(model, optimizer_str, lr_str, loss_str, alpha1=1, alpha2=1, alpha3=1, beta1=1, beta2=1, beta3=1, num_voxels=1, mimick=False):
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
        loss = dice_loss(0, K.epsilon())
        model.compile(loss=loss, metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)
    elif loss_str == 'cross_entropy':
        loss = cross_entropy_loss()
        model.compile(loss=loss, metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)
    elif loss_str == "mime":
        loss = mime_loss(alpha1 / num_voxels, beta1 / num_voxels, 
                         alpha2 / num_voxels, beta2 / num_voxels,
                         alpha3 / num_voxels, beta3 / num_voxels, mimick)
        model.compile(loss=loss, metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)

def cross_entropy_loss():
    def fn(y_true, y_pred):
        loss = 0.0
        for i in range(y_true.shape[3]):
            loss += K.mean(K.categorical_crossentropy(y_true[:, :, :, i], y_pred[:, :, :, i]))
        return loss / y_true.shape[3]
    return fn
         
def dice_coef_a(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(K.cast(y_true, tf.float32))
    y_pred_f = K.flatten(K.cast(y_pred, tf.float32))
    return - 2 * (mime_U(y_true_f, y_pred_f)  - mime_I(y_true_f, y_pred_f)) / (np.square(mime_U(y_true_f, y_pred_f)))

def dice_coef_b(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 2 * mime_I(y_true_f, y_pred_f) / (np.square(mime_U(y_true_f, y_pred_f)))

def mime_U(y, s):
    return (K.sum(y) + K.sum(s)) + K.epsilon()

def mime_I(y, s):
    return K.sum(y * s)

def dice_coef(y_true, y_pred, smooth_alpha=0, smooth_beta=K.epsilon()):
    y_true_f = K.flatten(K.cast(y_true, tf.float32))
    y_pred_f = K.flatten(K.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    dice = ((2. * intersection + smooth_alpha) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth_beta))
    return dice

def dice_loss(alpha=0, beta=K.epsilon()):
    def loss_fn(y_true, y_pred):
        loss = 0.0
        for i in range(np.shape(y_true)[3]):
            loss += 1 - dice_coef(y_true[:, :, :, i], y_pred[:, :, :, i], alpha, beta)
        return loss / y_true.shape[3]
    return loss_fn

def mime_loss_alpha(y_true, y_pred):
    mask_a = tf.not_equal(y_true, 0.0)
    loss_a = y_pred[mask_a]
    loss = tf.reduce_sum(loss_a)
    return - loss

def mime_loss_beta(y_true, y_pred):
    mask_b = tf.equal(y_true, 0.0)
    loss_b = y_pred[mask_b]
    loss = tf.reduce_sum(loss_b)
    return loss

def mime_loss(alpha1, alpha2, alpha3, beta1, beta2, beta3, mimick=False):
    import tensorflow as tf
    def loss_fn(y_true, y_pred):
        if mimick:
            alpha1 = dice_coef_a(y_true[:, :, :, 0], y_pred[:, :, :, 0])
            alpha2 = dice_coef_a(y_true[:, :, :, 1], y_pred[:, :, :, 1])
            alpha3 = dice_coef_a(y_true[:, :, :, 2], y_pred[:, :, :, 2])
            beta1 = dice_coef_b(y_true[:, :, :, 0], y_pred[:, :, :, 0])
            beta2 = dice_coef_b(y_true[:, :, :, 1], y_pred[:, :, :, 1])
            beta3 = dice_coef_b(y_true[:, :, :, 2], y_pred[:, :, :, 2])

        loss_0_a = y_pred[:, :, :, 0][tf.not_equal(y_true[:, :, :, 0], 0.0)]
        loss_0_b = y_pred[:, :, :, 0][tf.equal(y_true[:, :, :, 0], 0.0)]

        loss_1_a = y_pred[:, :, :, 1][tf.not_equal(y_true[:, :, :, 1], 0.0)]
        loss_1_b = y_pred[:, :, :, 1][tf.equal(y_true[:, :, :, 1], 0.0)]

        loss_2_a = y_pred[:, :, :, 2][tf.not_equal(y_true[:, :, :, 2], 0.0)]
        loss_2_b = y_pred[:, :, :, 2][tf.equal(y_true[:, :, :, 2], 0.0)]

        loss = - alpha1 * tf.reduce_sum(loss_0_a) + beta1 * tf.reduce_sum(loss_0_b)\
        - alpha2 * tf.reduce_sum(loss_1_a) + beta2 * tf.reduce_sum(loss_1_b)\
        - alpha3 * tf.reduce_sum(loss_2_a) + beta3 * tf.reduce_sum(loss_2_b)
        return loss / y_true.shape[3]
    return loss_fn

def evaluate(experiment, gen, model, name, labels, epoch):
    x_val, y_val = gen
    metric_dice = []
    metric_dice_a = []
    metric_dice_b = []
    metric_tp = []
    metric_tn = []
    metric_fp = []
    metric_fn = []
    for label in labels:
        metric_dice.append([])
        metric_dice_a.append([])
        metric_dice_b.append([])
        metric_tp.append([])
        metric_tn.append([])
        metric_fn.append([])
        metric_fp.append([])

    for patient in list(x_val.keys()):
        x = x_val[patient]
        y = y_val[patient]

        pred = np.zeros_like(y)
        for i in range(np.shape(x)[0]):
            if (np.max(x[i:i+1, :, :, :]) > 0):
                pred[i:i+1, :, :, :] = model.predict_on_batch(x[i:i+1, ])

        pred = np.array(pred)
        for j in range(np.shape(y)[3]):
            current_y = y[:, :, :, j].astype(np.float32)
            current_pred = pred[:, :, :, j].astype(np.float32)
            for i in range(np.shape(current_y)[2]):
                print(f"MIME_I_{labels[j]}: {mime_I(current_y[:, :, i], current_pred[:, :, i])}")
                print(f"MIME_U_{labels[j]}: {mime_U(current_y[:, :, i], current_pred[:, :, i])}")
                metric_dice_a[j].append(dice_coef_a(current_y[:, :, i], current_pred[:, :, i]).numpy())
                metric_dice_b[j].append(dice_coef_b(current_y[:, :, i], current_pred[:, :, i]).numpy())
            metric_dice[j].append(dice_coef(current_y, current_pred).numpy())
            metric_tp[j].append(np.sum((current_y == 1) * (current_pred >= 0.5)))
            metric_tn[j].append(np.sum((current_y == 0) * (current_pred < 0.5)))
            metric_fp[j].append(np.sum((current_y == 0) * (current_pred >= 0.5)))
            metric_fn[j].append(np.sum((current_y == 1) * (current_pred < 0.5)))

    for j in range(len(labels)):
        metric_dice[j] = np.array(metric_dice[j])
        metric_dice_a[j] = np.array(metric_dice_a[j])
        metric_dice_b[j] = np.array(metric_dice_b[j])
        print(f"{name} Dice {labels[j]}: {np.mean(np.mean(metric_dice[j]))}")
        experiment.log_metrics({f'{name}_dice_{labels[j]}': np.mean(metric_dice[j]),
                                f'{name}_dice_{labels[j]}_std': np.std(metric_dice[j]),
                                f'{name}_dice_a_{labels[j]}': np.mean(metric_dice_a[j]),
                                f'{name}_dice_a_{labels[j]}_std': np.std(metric_dice_a[j]),
                                f'{name}_dice_b_{labels[j]}': np.mean(metric_dice_b[j]),
                                f'{name}_dice_b_{labels[j]}_std': np.std(metric_dice_b[j]),
                                f'{name}_tp_{labels[j]}': np.mean(metric_tp[j]),
                                f'{name}_tn_{labels[j]}': np.mean(metric_tn[j]),
                                f'{name}_fp_{labels[j]}': np.mean(metric_fp[j]),
                                f'{name}_fn_{labels[j]}': np.mean(metric_fn[j]),
                                f'{name}_tp_{labels[j]}_std': np.std(metric_tp[j]),
                                f'{name}_tn_{labels[j]}_std': np.std(metric_tn[j]),
                                f'{name}_fp_{labels[j]}_std': np.std(metric_fp[j]),
                                f'{name}_fn_{labels[j]}_std': np.std(metric_fn[j])}, epoch=epoch)

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

def dice_squared_loss(y_true, y_pred, smooth=0.1):    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.square(y_true_f) + K.square(y_pred_f))
    difference = K.sum(K.square(y_true_f - y_pred_f))
    dice = ((difference / intersection) + difference / (tf.cast(tf.size(y_true_f), tf.float32)))
    return dice