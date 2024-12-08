def ssim_mae_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    mae = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=[1, 2, 3])
    return (1.0 - ssim) + mae

def msssim_mae_loss(y_true, y_pred):
    mae = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=[1, 2, 3])
    level = 5
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    msssim = []
    for _ in range(level):
        ssim_res = tf.image.ssim(y_true, y_pred, max_val=1.0)
        msssim.append(ssim_res)
        y_true = tf.nn.avg_pool2d(y_true, 2, 2, 'SAME')
        y_pred = tf.nn.avg_pool2d(y_pred, 2, 2, 'SAME')
    msssim = tf.reduce_sum(tf.stack(msssim, axis=1) * weights, axis=1)
    return (1.0 - msssim) + mae

def dft_loss(y_true, y_pred):
    dft_pred = tf.signal.rfft2d(y_pred)
    dft_true = tf.signal.rfft2d(y_true)
    dft_pred = tf.math.log(tf.math.abs(dft_pred) + 1e-12)
    dft_true = tf.math.log(tf.math.abs(dft_true) + 1e-12)
    dft_loss = tf.math.reduce_mean(tf.math.abs(dft_true - dft_pred), axis=[1, 2, 3])
    pix_loss = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.01 * dft_loss + pix_loss
    return loss

def dct2d(x):
    x = tf.squeeze(x, axis=-1)
    x_dct_width = tf.signal.dct(x, type=2, norm='ortho', axis=-1)
    x_transposed = tf.transpose(x_dct_width, perm=[0, 2, 1])
    x_dct_height = tf.signal.dct(x_transposed, type=2, norm='ortho', axis=-1)
    return x_dct_height

def dct_loss(y_true, y_pred):
    dct_pred = dct2d(y_pred)
    dct_true = dct2d(y_true)
    loss = tf.math.reduce_mean(tf.math.abs(dct_true - dct_pred), axis=[1, 2])
    return loss
