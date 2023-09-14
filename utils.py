from keras import backend as K
import tensorflow as tf
import numpy as np
import os

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)

def compile(model, dataset, optimizer_str, lr_str, loss_str, alpha1=1, alpha2=1, alpha3=1, alpha4=1, beta1=1, beta2=1, beta3=1, beta4=1, num_voxels=1, mimick=False):
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
        if (dataset == "WMH"):
            loss = mime_loss_wmh(alpha1, alpha2, alpha3,
                                 beta1, beta2, beta3, num_voxels)
        elif (dataset == "ACDC"):
            loss = mime_loss_acdc(alpha1, alpha2, alpha3, alpha4,
                                  beta1, beta2, beta3, beta4, num_voxels)
        model.compile(loss=loss, metrics=[mime_loss_alpha, mime_loss_beta], optimizer=optimizer)

def cross_entropy_loss():
    def fn(y_true, y_pred):
        loss = 0.0
        for i in range(y_true.shape[3]):
            loss += K.mean(K.categorical_crossentropy(y_true[:, :, :, i], y_pred[:, :, :, i]))
        return loss / y_true.shape[3]
    return fn
         
def dice_coef_a(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return - 2 * (mime_U(y_true_f, y_pred_f)  - mime_I(y_true_f, y_pred_f)) / (mime_U(y_true_f, y_pred_f)**2)

def dice_coef_b(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return 2 * mime_I(y_true_f, y_pred_f) / (mime_U(y_true_f, y_pred_f)**2)

def mime_U(y, s):
    return (K.sum(y) + K.sum(s)) + K.epsilon()

def mime_I(y, s):
    return K.sum(y * s)

def dice_coef(y_true, y_pred, smooth_alpha=0, smooth_beta=K.epsilon()):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) + smooth_beta
    dice = ((2. * intersection) / union)
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
    loss = K.sum(loss_a)
    return - loss

def mime_loss_beta(y_true, y_pred):
    mask_b = tf.equal(y_true, 0.0)
    loss_b = y_pred[mask_b]
    loss = K.sum(loss_b)
    return loss

def mime_loss_wmh(alpha_1, alpha_2, alpha_3, beta_1, beta_2, beta_3, num_voxels):
    import tensorflow as tf
    replace_alpha1 = False
    replace_alpha2 = False
    replace_alpha3 = False

    replace_beta1 = False
    replace_beta2 = False
    replace_beta3 = False

    alpha1 = alpha_1
    if (alpha1 == "-"):
        replace_alpha1 = True

    alpha2 = alpha_2
    if (alpha2 == "-"):
        replace_alpha2 = True

    alpha3 = alpha_3
    if (alpha3 == "-"):
        replace_alpha3 = True

    beta1 = beta_1
    if (beta1 == "-"):
        replace_beta1 = True

    beta2 = beta_2
    if (beta2 == "-"):
        replace_beta2 = True
        
    beta3 = beta_3
    if (beta3 == "-"):
        replace_beta3 = True

    def loss_fn(y_true, y_pred):
        if (replace_alpha1):
            alpha1 = - dice_coef_a(y_true[:, :, :, 0], y_pred[:, :, :, 0])
        else:
            alpha1 = alpha_1 / num_voxels

        if (replace_alpha2):
            alpha2 = - dice_coef_a(y_true[:, :, :, 1], y_pred[:, :, :, 1])
        else:
            alpha2 = alpha_2 / num_voxels

        if (replace_alpha3):
            alpha3 = - dice_coef_a(y_true[:, :, :, 2], y_pred[:, :, :, 2])
        else:
            alpha3 = alpha_3 / num_voxels

        if (replace_beta1):
            beta1 = dice_coef_b(y_true[:, :, :, 0], y_pred[:, :, :, 0])
        else:
            beta1 = beta_1 / num_voxels

        if (replace_beta2):
            beta2 = dice_coef_b(y_true[:, :, :, 1], y_pred[:, :, :, 1])
        else:
            beta2 = beta_2 / num_voxels
        
        if (replace_beta3):
            beta3 = dice_coef_b(y_true[:, :, :, 2], y_pred[:, :, :, 2])
        else:
            beta3 = beta_3 / num_voxels

        loss_0_a = y_pred[:, :, :, 0][tf.not_equal(y_true[:, :, :, 0], 0.0)]
        loss_0_b = y_pred[:, :, :, 0][tf.equal(y_true[:, :, :, 0], 0.0)]

        loss_1_a = y_pred[:, :, :, 1][tf.not_equal(y_true[:, :, :, 1], 0.0)]
        loss_1_b = y_pred[:, :, :, 1][tf.equal(y_true[:, :, :, 1], 0.0)]

        loss_2_a = y_pred[:, :, :, 2][tf.not_equal(y_true[:, :, :, 2], 0.0)]
        loss_2_b = y_pred[:, :, :, 2][tf.equal(y_true[:, :, :, 2], 0.0)]

        loss = - alpha1 * K.sum(loss_0_a) + beta1 * K.sum(loss_0_b)\
        - alpha2 * K.sum(loss_1_a) + beta2 * K.sum(loss_1_b)\
        - alpha3 * K.sum(loss_2_a) + beta3 * K.sum(loss_2_b)
        return 1 + (loss / y_true.shape[3])
    return loss_fn

def mime_loss_acdc(alpha_1, alpha_2, alpha_3, alpha_4, beta_1, beta_2, beta_3, beta_4, num_voxels):
    import tensorflow as tf
    replace_alpha1 = False
    replace_alpha2 = False
    replace_alpha3 = False
    replace_alpha4 = False

    replace_beta1 = False
    replace_beta2 = False
    replace_beta3 = False
    replace_beta4 = False

    alpha1 = alpha_1
    if (alpha1 == "-"):
        replace_alpha1 = True

    alpha2 = alpha_2
    if (alpha2 == "-"):
        replace_alpha2 = True

    alpha3 = alpha_3
    if (alpha3 == "-"):
        replace_alpha3 = True

    alpha4 = alpha_4
    if (alpha4 == "-"):
        replace_alpha4 = True

    beta1 = beta_1
    if (beta1 == "-"):
        replace_beta1 = True

    beta2 = beta_2
    if (beta2 == "-"):
        replace_beta2 = True
        
    beta3 = beta_3
    if (beta3 == "-"):
        replace_beta3 = True

    beta4 = beta_4
    if (beta4 == "-"):
        replace_beta4 = True

    def loss_fn(y_true, y_pred):
        if (replace_alpha1):
            alpha1 = - dice_coef_a(y_true[:, :, :, 0], y_pred[:, :, :, 0])
        else:
            alpha1 = alpha_1 / num_voxels

        if (replace_alpha2):
            alpha2 = - dice_coef_a(y_true[:, :, :, 1], y_pred[:, :, :, 1])
        else:
            alpha2 = alpha_2 / num_voxels

        if (replace_alpha3):
            alpha3 = - dice_coef_a(y_true[:, :, :, 2], y_pred[:, :, :, 2])
        else:
            alpha3 = alpha_3 / num_voxels

        if (replace_alpha4):
            alpha4 = - dice_coef_a(y_true[:, :, :, 3], y_pred[:, :, :, 3])
        else:
            alpha4 = alpha_4 / num_voxels

        if (replace_beta1):
            beta1 = dice_coef_b(y_true[:, :, :, 0], y_pred[:, :, :, 0])
        else:
            beta1 = beta_1 / num_voxels

        if (replace_beta2):
            beta2 = dice_coef_b(y_true[:, :, :, 1], y_pred[:, :, :, 1])
        else:
            beta2 = beta_2 / num_voxels
        
        if (replace_beta3):
            beta3 = dice_coef_b(y_true[:, :, :, 2], y_pred[:, :, :, 2])
        else:
            beta3 = beta_3 / num_voxels

        if (replace_beta4):
            beta4 = dice_coef_b(y_true[:, :, :, 3], y_pred[:, :, :, 3])
        else:
            beta4 = beta_4 / num_voxels

        loss_0_a = y_pred[:, :, :, 0][tf.not_equal(y_true[:, :, :, 0], 0.0)]
        loss_0_b = y_pred[:, :, :, 0][tf.equal(y_true[:, :, :, 0], 0.0)]

        loss_1_a = y_pred[:, :, :, 1][tf.not_equal(y_true[:, :, :, 1], 0.0)]
        loss_1_b = y_pred[:, :, :, 1][tf.equal(y_true[:, :, :, 1], 0.0)]

        loss_2_a = y_pred[:, :, :, 2][tf.not_equal(y_true[:, :, :, 2], 0.0)]
        loss_2_b = y_pred[:, :, :, 2][tf.equal(y_true[:, :, :, 2], 0.0)]

        loss_3_a = y_pred[:, :, :, 3][tf.not_equal(y_true[:, :, :, 3], 0.0)]
        loss_3_b = y_pred[:, :, :, 3][tf.equal(y_true[:, :, :, 3], 0.0)]

        loss = - alpha1 * K.sum(loss_0_a) + beta1 * K.sum(loss_0_b)\
        - alpha2 * K.sum(loss_1_a) + beta2 * K.sum(loss_1_b)\
        - alpha3 * K.sum(loss_2_a) + beta3 * K.sum(loss_2_b)\
        - alpha4 * K.sum(loss_3_a) + beta4 * K.sum(loss_3_b)
        return 1 + (loss / y_true.shape[3])
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