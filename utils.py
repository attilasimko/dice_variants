from sympy import N
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
import os
from losses import dice_loss, cross_entropy_loss, coin_loss, dice_ce_loss, get_coeffs, dice_coef, coin_coef_a, coin_coef_b, squared_dice_loss

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

    # tf.compat.v1.enable_eager_execution()
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # K.set_session(sess)

def model_compile(model, optimizer_str, lr_str, loss_str, epsilon="1", alphas=["-"], betas=["-"]):
    import tensorflow

    weights = model.get_weights()
    if (epsilon == "-"):
        epsilon = K.epsilon()
    else:
        epsilon = float(epsilon)

    lr = float(lr_str)
    if optimizer_str == 'Adam':
        optimizer = tensorflow.keras.optimizers.Adam(lr, beta_1=0.99, beta_2=0.999)
    elif optimizer_str == 'SGD':
        optimizer = tensorflow.keras.optimizers.SGD(lr, momentum=0.9)
    elif optimizer_str == 'RMSprop':
        optimizer = tensorflow.keras.optimizers.RMSprop(lr)
    else:
        raise NotImplementedError
    
    
    if loss_str == 'dice':
        loss = dice_loss(epsilon)
    elif loss_str == 'dice_squared':
        loss = squared_dice_loss(epsilon)
    elif loss_str == 'mean_squared_error':
        loss = tf.losses.mean_squared_error
    elif loss_str == 'cross_entropy':
        loss = cross_entropy_loss()
    elif loss_str == "coin":
        loss = coin_loss(alphas, betas, epsilon)
        print("Coin loss using - " + str(alphas) + " - " + str(betas))
    elif loss_str == "dice+cross_entropy":
        loss = dice_ce_loss(epsilon)
    else:
        raise NotImplementedError
    
    model.compile(loss=loss, metrics=[coin_coef_a, coin_coef_b], optimizer=optimizer, run_eagerly=True)
    model.set_weights(weights)

def plot_results(gen_val, model, dataset, experiment, save_path):
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Model
    model = Model(model.inputs, [model.layers[-2].output, model.layers[-1].output])
    for idx in range(100):#len(gen_val)):
        x, y = gen_val.next_batch()

        if (np.sum(y[0, :, :, 1:]) == 0):
            continue

        pred = model.predict_on_batch(x)
        if (dataset == "WMH"):
            plt.subplot(351)
            plt.imshow(x[0, :, :, 0], cmap="gray", interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(352)
            plt.imshow(x[0, :, :, 1], cmap="gray", interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(353)
            plt.imshow(y[0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(354)
            plt.imshow(y[0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(355)
            plt.imshow(y[0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(358)
            plt.imshow(pred[0][0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(359)
            plt.imshow(pred[0][0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 10)
            plt.imshow(pred[0][0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(3, 5, 13)
            plt.imshow(pred[1][0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 14)
            plt.imshow(pred[1][0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 15)
            plt.imshow(pred[1][0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
        elif (dataset == "ACDC"):
            plt.subplot(351)
            plt.imshow(x[0, :, :, 0], cmap="gray", interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(352)
            plt.imshow(y[0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(353)
            plt.imshow(y[0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(354)
            plt.imshow(y[0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(355)
            plt.imshow(y[0, :, :, 3], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(357)
            plt.imshow(pred[0][0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(358)
            plt.imshow(pred[0][0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 9)
            plt.imshow(pred[0][0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 10)
            plt.imshow(pred[0][0, :, :, 3], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(3, 5, 12)
            plt.imshow(pred[1][0, :, :, 0], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 13)
            plt.imshow(pred[1][0, :, :, 1], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 14)
            plt.imshow(pred[1][0, :, :, 2], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 5, 15)
            plt.imshow(pred[1][0, :, :, 3], cmap="gray", vmin=0, vmax=1, interpolation="none")
            plt.xticks([])
            plt.yticks([])
        
        plt.savefig(save_path + str(idx) + ".png")
        plt.close()
        experiment.log_image(save_path + str(idx) + ".png", overwrite=False)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def simplex_etf(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))


def plot_model_insight(experiment, weights, save_path, name, epoch):
    import matplotlib.pyplot as plt
    l2_norms = []
    markers = []
    labels = []

    for w in weights:
        if w.ndim in (4, 5):  # conv kernel (4D) or dense kernel (2D)
            for i in range(w.shape[-1]):
                l2_norms.append(np.linalg.norm(w[..., i]))
                markers.append('o')
                labels.append('kernel')
        elif w.ndim in (1, 2):  # bias or BN params
            for i in range(w.shape[-1]):
                l2_norms.append(np.linalg.norm(w[..., i]))
                markers.append('x')
                labels.append('bias')

    plt.figure(figsize=(20, 10))
    for i, (norm, m, lbl) in enumerate(zip(l2_norms, markers, labels)):
        plt.scatter(i, norm, c='#1f77b4', marker=m, label=lbl if i == labels.index(lbl) else "", s=10)

    plt.plot(range(len(l2_norms)), l2_norms, linestyle='--', color="#74a8ce", alpha=0.4)
    plt.xlabel("Parameter index (layer-by-layer)")
    plt.ylabel("L2 norm")
    plt.title("Layer parameter norms")
    plt.legend()
    plt.savefig(save_path + name + ".png")
    plt.close()
    experiment.log_image(save_path + name + ".png", step=epoch)
    return

def train_model(model, skip_background, x, y):
    inp = tf.convert_to_tensor(x, dtype=tf.float64)
    with tf.GradientTape() as tape:
        predictions = model(inp)
        if (skip_background):
            loss_value = model.loss(tf.convert_to_tensor(y[..., 1:], tf.float64), predictions[..., 1:])  
        else:
            loss_value = model.loss(tf.convert_to_tensor(y, tf.float64), predictions)  
        
    grads = tape.gradient(loss_value, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return float(loss_value.numpy()), [np.array(grad.numpy().copy(), np.float16) for grad in grads]

def evaluate(experiment, gen, model, name, labels, epoch):
    import matplotlib.pyplot as plt
    save_path = experiment.get_parameter('save_path')

    # last_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("conv2d_13").output)
    # layer_outputs = [layer.output for layer in model.layers]
    # activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    x_val, y_val = gen
    metric_dice = []
    metric_dice_a = []
    metric_dice_b = []
    metric_u = []
    metric_i = []
    metric_conf_matrix = np.zeros((len(labels), len(labels)))
    etf = np.zeros((len(labels), len(labels)))
    features = []
    grads = []
    for _ in labels:
        metric_dice.append([])
        metric_dice_a.append([])
        metric_dice_b.append([])
        metric_u.append([])
        metric_i.append([])
        features.append([])

    for patient in list(x_val.keys()):
        x = x_val[patient]
        y = y_val[patient]
        
        pred = np.zeros_like(y)
        for idx in range(np.shape(x)[0]):
            if (np.max(x[idx:idx+1, :, :, :]) > 0):
                pred[idx:idx+1, :, :, :] = model.predict_on_batch(x[idx:idx+1, ])

        # last_layer_pred = last_layer_model.predict_on_batch(x)
        # for idx in range(np.shape(x)[0]):
        #     for pixel_i in range(np.shape(y)[1]):
        #         for pixel_j in range(np.shape(y)[2]):
        #             pixel_y = np.argmax(y[idx, pixel_i, pixel_j, :])
        #             features[pixel_y].append(np.ndarray.flatten(last_layer_pred[idx, pixel_i, pixel_j, :]))

        grad_patient = []
        pred = np.array(pred)
        for j in range(np.shape(y)[3]):
            current_y = y[:, :, :, j].astype(np.float64)
            current_pred = pred[:, :, :, j].astype(np.float64)

            metric_dice[j].append(dice_coef(current_y, current_pred).numpy())
            
            if (name != "train"):
                grad = []
                for slc in range(np.shape(current_y)[0]):
                    I, U, a, b = get_coeffs(current_y[slc, :, :], current_pred[slc, :, :])
                    grad.append([I, U])
                    metric_dice_a[j].append(a)
                    metric_dice_b[j].append(b)
                    metric_u[j].append(a)
                    metric_i[j].append(b)
                grad_patient.append([np.array(grad)[:, 0], np.array(grad)[:, 1]])

            for i in range(np.shape(y)[3]):
                metric_conf_matrix[j, i] += np.sum(y[:, :, :, j].astype(np.float64) * pred[:, :, :, i].astype(np.float64))
        grads.append(np.vstack(grad_patient))
                
    # gamma_c = [np.mean(feature, 0) for feature in features]
    # gamma_g = np.mean(gamma_c, 0)

    # NC1 = [np.mean([np.dot(feat - gamma, feat - gamma) for feat in feature]) for feature, gamma in zip(features, gamma_c)]
    # NC1_std = [np.std([np.dot(feat - gamma, feat - gamma) for feat in feature]) for feature, gamma in zip(features, gamma_c)]
    # NC2 = np.zeros((len(labels), len(labels)))
    
    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         etf[i, j] = simplex_etf(gamma_c[i] - gamma_g, gamma_c[j] - gamma_g)
    #         NC2[i, j] = np.abs(np.sum(np.abs(gamma_c[i] - gamma_g)) - np.sum(np.abs(gamma_c[j] - gamma_g)))

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
                                # f'{name}_nc1_{labels[j]}': NC1[j],
                                # f'{name}_nc1_{labels[j]}_std': NC1_std[j],
                                # f'{name}_nc2_{labels[j]}': np.mean(np.delete(NC2[j, :], j)),
                                # f'{name}_nc2_{labels[j]}_std': np.std(np.delete(NC2[j, :], j)),
                                # f'{name}_etf_{labels[j]}': np.mean(np.delete(etf[j, :], j)),
                                # f'{name}_etf_{labels[j]}_std': np.std(np.delete(etf[j, :], j)),
                                # f'{name}_gamma_g': gamma_g,
                                # f'{name}_gamma_c_{labels[j]}': gamma_c[j],
                                f'{name}_u_{labels[j]}': np.mean(metric_u[j]),
                                f'{name}_i_{labels[j]}': np.mean(metric_i[j]),
                                f'{name}_u_{labels[j]}_std': np.std(metric_u[j]),
                                f'{name}_i_{labels[j]}_std': np.std(metric_i[j])}, epoch=epoch)
        
    experiment.log_confusion_matrix(matrix=metric_conf_matrix, labels=labels, epoch=epoch, file_name='metric_conf.json')
    # experiment.log_confusion_matrix(matrix=etf, labels=labels, epoch=epoch, file_name='NC2_1.json')
    # experiment.log_confusion_matrix(matrix=NC2, labels=labels, epoch=epoch, file_name='NC2_2.json')
    experiment.log_metrics({f'{name}_avg_dice': np.mean(np.mean(metric_dice))}, epoch=epoch)
    plt.savefig(save_path + "coefs.png")
    plt.close()
    experiment.log_image(save_path + "coefs.png", step=epoch)

    return np.reshape(np.hstack(grads).T, (-1))