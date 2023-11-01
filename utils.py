from keras import backend as K
import tensorflow as tf
import numpy as np
import os

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)

def compile(model, optimizer_str, lr_str, loss_str, skip_background, epsilon="1", alphas=["-"], betas=["-"]):
    import tensorflow

    if (epsilon == "-"):
        epsilon = K.epsilon()
    else:
        epsilon = float(epsilon)

    lr = float(lr_str)
    if optimizer_str == 'Adam':
        optimizer = tensorflow.keras.optimizers.Adam(lr)
    elif optimizer_str == 'SGD':
        optimizer = tensorflow.keras.optimizers.SGD(lr, momentum=0.9)
    elif optimizer_str == 'RMSprop':
        optimizer = tensorflow.keras.optimizers.RMSprop(lr)
    else:
        raise NotImplementedError
    
    if loss_str == 'dice':
        loss = dice_loss(skip_background, epsilon)
    elif loss_str == 'cross_entropy':
        loss = cross_entropy_loss(skip_background)
    elif loss_str == 'squared_dice':
        loss = squared_dice_loss(skip_background)
    elif loss_str == "coin":
        loss = coin_loss(alphas, betas, epsilon)
    else:
        raise NotImplementedError
    
    model.compile(loss=loss, metrics=[coin_coef_a, coin_coef_b], optimizer=optimizer, run_eagerly=True)

def cross_entropy_loss(skip_background=False):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        for slc in range(y_true.shape[0]):
            loss += tf.losses.categorical_crossentropy(y_true[slc, :, :, start_idx:], y_pred[slc, :, :, start_idx:])
        return loss / y_true.shape[0]
    return loss_fn

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

def coin_b(u, i):
    return 2 * i / u**2

def coin_a(u):
    return - 2 / u

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

def dice_loss(skip_background=False, epsilon=1):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        for slc in range(y_true.shape[0]):
            for i in range(start_idx, y_true.shape[3]):
                loss += 1 - dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
        return loss / y_true.shape[0]
    return loss_fn

def squared_dice_loss(skip_background=False, epsilon=1):
    def loss_fn(y_true, y_pred):
        start_idx = 1 if skip_background else 0
        loss = 0.0
        for slc in range(y_true.shape[0]):
            for i in range(start_idx, y_true.shape[3]):
                loss += 1 - squared_dice_coef(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)
        return loss / y_true.shape[0]
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
        loss = 0.0
        for slc in range(y_true.shape[0]):
            total = 0.0
            for i in range(y_true.shape[3]):
                total += tf.stop_gradient(coin_coef_b(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon))

            for i in range(y_true.shape[3]):
                if (replace_alphas[i]):
                    alpha = tf.stop_gradient(coin_coef_a(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon)) #  * -tf.stop_gradient(tf.math.log(tf.where(tf.equal(y_pred[slc, :, :, i], 0.0), np.finfo(float).eps, y_pred[slc, :, :, i])))
                else:
                    alpha = float(alphas[i])

                if (replace_betas[i]):
                    beta = tf.stop_gradient(coin_coef_b(y_true[slc, :, :, i], y_pred[slc, :, :, i], epsilon))
                else:
                    beta = float(betas[i])

                loss += 1 + K.sum(alpha * y_true[slc, :, :, i] * y_pred[slc, :, :, i] + beta * y_pred[slc, :, :, i])
        return loss / y_true.shape[0]
    return loss_fn

def plot_grad(x, y, model, idx):
    import matplotlib.pyplot as plt

    coin_fn = coin_loss(["-", "-", "-", "-"], ["-", "-", "-", "-"])
    dice_fn = dice_loss()
    inp = tf.Variable(x[0:1, :, :, :], dtype=tf.float64)
    with tf.GradientTape() as tape:
        preds = model(inp)
        loss = coin_fn(tf.Variable(y[0:1, :, :, :], dtype=tf.float64), preds)   
    coin_grads = tape.gradient(loss, preds)
    with tf.GradientTape() as tape:
        preds = model(inp)
        loss = dice_fn(tf.Variable(y[0:1, :, :, :], dtype=tf.float64), preds)   
    dice_grads = tape.gradient(loss, preds)

    if (np.max(np.abs(coin_grads - dice_grads)) < 1e-20):
        return
    
    plt.figure(figsize=(15, 12))
    for i in range(np.shape(y)[3]):
        plt.subplot(4, 5, (i * 5) + 1)
        plt.imshow(y[0, :, :, i], cmap="gray", interpolation="none")
        plt.colorbar()
        plt.subplot(4, 5, (i * 5) + 2)
        plt.imshow(preds[0, :, :, i], cmap="gray", interpolation="none")
        plt.colorbar()
        plt.subplot(4, 5, (i * 5) + 3)
        plt.imshow(coin_grads[0, :, :, i], cmap="gray", interpolation="none")
        plt.colorbar()
        plt.subplot(4, 5, (i * 5) + 4)
        plt.imshow(dice_grads[0, :, :, i], cmap="gray", interpolation="none")
        plt.colorbar()
        plt.subplot(4, 5, (i * 5) + 5)
        plt.imshow(coin_grads[0, :, :, i] - dice_grads[0, :, :, i], cmap="coolwarm", interpolation="none")
        plt.colorbar()
        
    plt.savefig(f"figs/grads_{idx}_{str(np.max(np.abs(coin_grads - dice_grads)))}.png")
    plt.close()  

def evaluate(experiment, gen, model, name, labels, epoch):
    import matplotlib.pyplot as plt
    save_path = experiment.get_parameter('save_path')
    
    x_val, y_val = gen
    metric_dice = []
    metric_dice_a = []
    metric_dice_b = []
    metric_u = []
    metric_i = []
    metric_tp = []
    metric_tn = []
    metric_fp = []
    metric_fn = []
    grads = []
    for _ in labels:
        metric_dice.append([])
        metric_dice_a.append([])
        metric_dice_b.append([])
        metric_u.append([])
        metric_i.append([])
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

        grad_patient = []
        pred = np.array(pred)
        for j in range(np.shape(y)[3]):
            current_y = y[:, :, :, j].astype(np.float64)
            current_pred = pred[:, :, :, j].astype(np.float64)

            metric_dice[j].append(dice_coef(current_y, current_pred).numpy())
            
            if (name != "train"):
                grad = []
                for i in range(np.shape(current_y)[0]):
                    # if (np.sum(current_y[i, :, :]) > 0):
                    metric_dice_a[j].append(coin_coef_a(current_y[i, :, :], current_pred[i, :, :]).numpy())
                    metric_dice_b[j].append(coin_coef_b(current_y[i, :, :], current_pred[i, :, :]).numpy())
                    grad.append([coin_coef_a(current_y[i, :, :], current_pred[i, :, :]).numpy(),
                                coin_coef_b(current_y[i, :, :], current_pred[i, :, :]).numpy()])
                    metric_u[j].append(coin_U(current_y[i, :, :], current_pred[i, :, :]).numpy())
                    metric_i[j].append(coin_I(current_y[i, :, :], current_pred[i, :, :]).numpy())
                metric_tp[j].append(np.sum((current_y == 1) * (current_pred >= 0.5)))
                metric_tn[j].append(np.sum((current_y == 0) * (current_pred < 0.5)))
                metric_fp[j].append(np.sum((current_y == 0) * (current_pred >= 0.5)))
                metric_fn[j].append(np.sum((current_y == 1) * (current_pred < 0.5)))

                grad_patient.append([np.array(grad)[:, 0], np.array(grad)[:, 1]])

        grads.append(np.vstack(grad_patient))
                
    
    plt.figure(figsize=(12, int(len(labels) * 4)))
    for j in range(len(labels)):
        if (name != "train"):
            try:
                plt.subplot(len(labels), 3, (j * 3) + 1)
                plt.hist(metric_dice_a[j])
                plt.title(f"{name} {labels[j]}_coef_a")
                plt.subplot(len(labels), 3, (j * 3) + 2)
                plt.hist(metric_dice_b[j])
                plt.title(f"{name} {labels[j]}_coef_b")
                plt.subplot(len(labels), 3, (j * 3) + 3)
                plt.hist(np.array(metric_dice_b[j]) / np.array(metric_dice_a[j]))
                plt.title(f"{name} {labels[j]}_coef_ratio_(b/a)")
            except:
                print("An exception occured")

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
                                f'{name}_u_{labels[j]}': np.mean(metric_u[j]),
                                f'{name}_i_{labels[j]}': np.mean(metric_i[j]),
                                f'{name}_u_{labels[j]}_std': np.std(metric_u[j]),
                                f'{name}_i_{labels[j]}_std': np.std(metric_i[j]),
                                f'{name}_tp_{labels[j]}': np.mean(metric_tp[j]),
                                f'{name}_tn_{labels[j]}': np.mean(metric_tn[j]),
                                f'{name}_fp_{labels[j]}': np.mean(metric_fp[j]),
                                f'{name}_fn_{labels[j]}': np.mean(metric_fn[j]),
                                f'{name}_tp_{labels[j]}_std': np.std(metric_tp[j]),
                                f'{name}_tn_{labels[j]}_std': np.std(metric_tn[j]),
                                f'{name}_fp_{labels[j]}_std': np.std(metric_fp[j]),
                                f'{name}_fn_{labels[j]}_std': np.std(metric_fn[j])}, epoch=epoch)

    experiment.log_metrics({f'{name}_avg_dice': np.mean(np.mean(metric_dice))}, epoch=epoch)
    plt.savefig(save_path + "coefs.png")
    plt.close()
    experiment.log_image(save_path + "coefs.png", step=epoch)

    return np.hstack(grads)

def boundary_loss(y_true, y_pred):
    raise NotImplementedError

def dice_squared_loss(y_true, y_pred, epsilon=0.1):    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.square(y_true_f) + K.square(y_pred_f) + epsilon)
    difference = K.sum(K.square(y_true_f - y_pred_f))
    dice = ((difference / intersection) + difference / (tf.cast(tf.size(y_true_f), tf.float64)))
    return dice