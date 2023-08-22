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
        model.compile(loss=loss, metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)
    elif loss_str == 'cross_entropy':
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(loss=loss, metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)
    elif loss_str == "mime":
        model.compile(loss=[mime_loss_alpha, mime_loss_beta], loss_weights=[alpha, beta], metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)

def dice_coef_a(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((intersection + smooth) / (K.sum(y_true_f) + smooth))

def dice_coef_b(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return ((K.sum(y_pred_f) + smooth) / (K.sum(y_true_f) + smooth))

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

def mime_loss_alpha(y_true, y_pred):
    loss = 0.0
    num_el = 0.0
    for slc in range(np.shape(y_true)[0]):
        mask_a = tf.not_equal(y_true[slc, :, :, :], 0.0)
        loss_a = y_pred[slc, :, :, :][mask_a]
        if (~tf.math.is_nan(tf.reduce_mean(loss_a))):
            loss += tf.reduce_mean(loss_a)
            num_el += 1
    return - loss / num_el

def mime_loss_beta(y_true, y_pred):
    loss = 0.0
    num_el = 0.0
    for slc in range(np.shape(y_true)[0]):
        mask_b = tf.equal(y_true[slc, :, :, :], 0.0)
        loss_b = y_pred[slc, :, :, :][mask_b]
        if (~tf.math.is_nan(tf.reduce_mean(loss_b))):
            loss += tf.reduce_mean(loss_b)
            num_el += 1
    return loss / num_el

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
            metric_dice[j].append(dice_coef(current_y, current_pred).numpy())
            metric_dice_a[j].append(dice_coef_a(current_y, current_pred).numpy())
            metric_dice_b[j].append(dice_coef_b(current_y, current_pred).numpy())
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