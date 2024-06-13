# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/utils.py

import sys
import numpy as np
import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.activations import softmax
from typing import Callable, Union

def bce_vol(y_true, y_pred, batch_size=16):
    bce = keras.losses.BinaryCrossentropy()
    vol_y_true_ml = tf.math.reduce_sum(y_true[:,:,:,1]) / 1000
    vol_y_pred_ml = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,1], 0.5), tf.float32)) / 1000
    err_vol = vol_y_true_ml - vol_y_pred_ml
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))
    return bce(y_true, y_pred) + tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

def vol(y_true, y_pred, batch_size=16):
    vol_y_true_ml = tf.math.reduce_sum(y_true[:,:,:,1]) / 1000
    vol_y_pred_ml = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,1], 0.5), tf.float32)) / 1000
    err_vol = vol_y_true_ml - vol_y_pred_ml
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))
    return tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

def bce_vol_3D(y_true, y_pred, batch_size=16):
    bce = keras.losses.BinaryCrossentropy()
    vol_y_true_ml = tf.math.reduce_sum(y_true[:,:,:,:,1]) / 1000
    vol_y_pred_ml = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,:,1], 0.75), tf.float32)) / 1000
    err_vol = vol_y_true_ml - vol_y_pred_ml
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))

    return bce(y_true, y_pred) + tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

def vol_3D(y_true, y_pred, batch_size=16):
    vol_y_true_ml = tf.math.reduce_sum(y_true[:,:,:,:,1]) / 1000
    vol_y_pred_ml = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,:,1], 0.5), tf.float32)) / 1000
    err_vol = vol_y_true_ml - vol_y_pred_ml
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))

    return tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

def fcl_vol_3D(y_true, y_pred, batch_size=16):
    fcl = binary_focal_loss()
    vol_y_true_ml = tf.math.reduce_sum(y_true[:,:,:,:,1]) / 1000
    vol_y_pred_ml = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,:,1], 0.75), tf.float32)) / 1000
    err_vol = vol_y_true_ml - vol_y_pred_ml
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))

    return fcl(y_true, y_pred) + tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

def bce_vol_3lbl(y_true, y_pred, batch_size=16):
    bce = keras.losses.BinaryCrossentropy()
    vol_y_true_ml = (tf.math.reduce_sum(y_true[:,:,:,2]) + tf.math.reduce_sum(y_true[:,:,:,3])) / 1000
    vol_y_pred_ml_grw = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,2], 0.25), tf.float32)) / 1000
    vol_y_pred_ml_stb = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,3], 0.25), tf.float32)) / 1000
    err_vol = vol_y_true_ml - (vol_y_pred_ml_grw + vol_y_pred_ml_stb)
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))
    return bce(y_true, y_pred) + tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

def vol_3lbl(y_true, y_pred, batch_size=16):
    vol_y_true_ml = (tf.math.reduce_sum(y_true[:,:,:,2]) + tf.math.reduce_sum(y_true[:,:,:,3])) / 1000
    vol_y_pred_ml_grw = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,2], 0.25), tf.float32)) / 1000
    vol_y_pred_ml_stb = tf.math.reduce_sum(tf.dtypes.cast(tf.math.greater_equal(y_pred[:,:,:,3], 0.25), tf.float32)) / 1000
    err_vol = vol_y_true_ml - (vol_y_pred_ml_grw + vol_y_pred_ml_stb)
    sq_err_vol = tf.math.reduce_mean(tf.pow((err_vol), 2))
    return tf.dtypes.cast(sq_err_vol / batch_size, tf.float32)

# Calculate Dice coefficient score
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_numpy(y_true, y_pred, smooth=1e-7):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

# Calculate Dice coefficient loss
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def squared_dice_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred)) ** 2

def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1., square_nominator=False, square_denom=False):
    axes = tuple(range(2, len(net_output.size())))
    if square_nominator:
        intersect = sum_tensor((net_output * gt) ** 2, axes, keepdim=False)
    else:
        intersect = sum_tensor(net_output * gt, axes, keepdim=False)

    if square_denom:
        denom = sum_tensor(net_output ** 2 + gt ** 2, axes, keepdim=False)
    else:
        denom = sum_tensor(net_output + gt, axes, keepdim=False)

    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result

# https://github.com/maxvfischer/keras-image-segmentation-loss-functions
def multiclass_weighted_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted Dice loss.
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def cat_weighted_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute weighted Dice loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, backend.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * backend.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights # Broadcasting
        denominator = backend.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return cat_weighted_dice_loss

# https://github.com/maxvfischer/keras-image-segmentation-loss-functions
def multiclass_weighted_squared_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor],
                                                                                                   tf.Tensor]:
    """
    Weighted squared Dice loss.
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted squared Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def cat_weighted_squared_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute weighted squared Dice loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted squared Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, backend.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * backend.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2) * class_weights  # Broadcasting
        denominator = backend.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return cat_weighted_squared_dice_loss

# https://github.com/maxvfischer/keras-image-segmentation-loss-functions
def multiclass_weighted_cross_entropy(class_weights: list, is_logits: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Multi-class weighted cross entropy.
        WCE(p, p̂) = −Σp*log(p̂)*class_weights
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Weight coefficients (list of floats)
    :param is_logits: If y_pred are logits (bool)
    :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the weighted cross entropy.
        :param y_true: Ground truth (tf.Tensor, shape=(None, None, None, None))
        :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        assert len(class_weights) == y_pred.shape[-1], f"Number of class_weights ({len(class_weights)}) needs to be the same as number " \
                                                 f"of classes ({y_pred.shape[-1]})"

        if is_logits:
            y_pred = softmax(y_pred, axis=-1)

        y_pred = backend.clip(y_pred, backend.epsilon(), 1-backend.epsilon())  # To avoid unwanted behaviour in backend.log(y_pred)

        # p * log(p̂) * class_weights
        wce_loss = y_true * backend.log(y_pred) * class_weights

        # Average over each data point/image in batch
        axis_to_reduce = range(1, backend.ndim(wce_loss))
        wce_loss = backend.mean(wce_loss, axis=axis_to_reduce)

        return -wce_loss

    return loss

# https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def bin_focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = backend.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = backend.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(backend.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = backend.ones_like(y_true) * alpha
        alpha_t = tf.where(backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -backend.log(p_t)
        weight = alpha_t * backend.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = backend.mean(backend.sum(loss, axis=1))
        return loss

    return bin_focal_loss

# https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def cat_focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return backend.mean(backend.sum(loss, axis=-1))

    return cat_focal_loss

def loss_for_2branches():

    def loss_2branches(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """        
        y_true_1 = y_true[0]
        y_true_2 = y_true[1]

        y_pred_1 = y_pred[0]
        y_pred_2 = y_pred[1]

        bfcl = tfa.losses.SigmoidFocalCrossEntropy()

        loss_l1 = bfcl(y_true_1, y_pred_1)
        loss_l2 = bfcl(y_true_2, y_pred_2)

        return (loss_l1 + loss_l2) / 2

    return loss_2branches


def loss_for_3branches_v2(alpha):

    # len_alpha = len(alpha)
    alpha = np.array(alpha, dtype=np.float32)

    def loss_3branches(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        y_true_1 = y_true[0]
        y_true_2 = y_true[1]
        y_true_3 = y_true[2]

        y_pred_1 = y_pred[0]
        y_pred_2 = y_pred[1]
        y_pred_3 = y_pred[2]

        bfcl = tfa.losses.SigmoidFocalCrossEntropy()
        fcl = categorical_focal_loss(alpha)

        loss_l1 = bfcl(y_true_1, y_pred_1)
        loss_l2 = bfcl(y_true_2, y_pred_2)
        loss_l3 = fcl(y_true_3, y_pred_3)
        avg = loss_l1 + loss_l2 + loss_l3

        return avg

    return loss_3branches

def loss_for_3branches(alpha):

    # len_alpha = len(alpha)
    alpha = np.array(alpha, dtype=np.float32)

    def loss_3branches(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        y_true_1 = y_true[0]
        y_true_2 = y_true[1]
        y_true_3 = y_true[2]

        y_pred_1 = y_pred[0]
        y_pred_2 = y_pred[1]
        y_pred_3 = y_pred[2]

        bfcl = tfa.losses.SigmoidFocalCrossEntropy()
        fcl = categorical_focal_loss(alpha)

        loss_l1 = backend.mean(backend.sum(bfcl(y_true_1, y_pred_1), axis=-1))
        loss_l2 = backend.mean(backend.sum(bfcl(y_true_2, y_pred_2), axis=-1))
        loss_l3 = fcl(y_true_3, y_pred_3)
        avg = (loss_l1 + loss_l2 + loss_l3) / 3

        return avg

    return loss_3branches

def mseVol_categorical_focal_loss(alpha, gamma=2.):

    # len_alpha = len(alpha)
    alpha = np.array(alpha, dtype=np.float32)

    def mseVol_cat_focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        focal = backend.mean(backend.sum(loss, axis=-1))

        # n_data = len_alpha
        [N, H, W, C] = tf.shape(y_true).numpy()

        # labels for 'y_true'
        y_true_flat = tf.reshape(y_true, [N * H * W, C])
        y_true_n_class = tf.argmax(y_true_flat, axis=1)
        y_true_n_class = tf.squeeze(tf.reshape(y_true_n_class, [N, H, W, 1]))

        # labels for 'y_pred'
        y_pred_flat = tf.reshape(y_pred, [N * H * W, C])
        y_pred_n_class = tf.argmax(y_pred_flat, axis=1)
        y_pred_n_class = tf.squeeze(tf.reshape(y_pred_n_class, [N, H, W, 1]))

        y_true_vol = tf.cast(backend.greater(y_true_n_class,1), tf.uint8) / 1000
        y_pred_vol = tf.cast(backend.greater(y_pred_n_class,1), tf.uint8) / 1000

        mse_vol = tf.reduce_sum(tf.math.square(y_true_vol - y_pred_vol)) / N

        return mse_vol * focal

    return mseVol_cat_focal_loss

def mse_vol_loss():
    def mseVol_loss(y_true, y_pred):
        [N, H, W, C] = tf.shape(y_true).numpy()

        # labels for 'y_true'
        y_true_flat = tf.reshape(y_true, [N * H * W, C])
        y_true_n_class = tf.argmax(y_true_flat, axis=1)
        y_true_n_class = tf.squeeze(tf.reshape(y_true_n_class, [N, H, W, 1]))

        # labels for 'y_pred'
        y_pred_flat = tf.reshape(y_pred, [N * H * W, C])
        y_pred_n_class = tf.argmax(y_pred_flat, axis=1)
        y_pred_n_class = tf.squeeze(tf.reshape(y_pred_n_class, [N, H, W, 1]))

        y_true_vol = tf.cast(backend.greater(y_true_n_class,1), tf.uint8) / 1000
        y_pred_vol = tf.cast(backend.greater(y_pred_n_class,1), tf.uint8) / 1000

        mse_vol = tf.reduce_sum(tf.math.square(y_true_vol - y_pred_vol)) / N

        return mse_vol
    return mseVol_loss

def convert_from_1hot_tf(label, to_float=False, batch_size=16, num_class=4, height=256, width=256):
    N = batch_size
    # _, H, W, C = label.shape
    H = height
    W = width
    C = num_class

    label_flat = tf.reshape(label, [N * H * W, C])
    n_data = N * H * W

    if to_float:
        label_n_class = tf.zeros((n_data, 1), tf.float32)
        max_class = tf.math.argmax(label_flat, axis=1)
        label_n_class = tf.cast(label_flat[range(n_data), max_class], tf.float32)
    else:
        label_n_class = tf.cast(tf.math.argmax(label_flat, axis=1), tf.int32)

    label_n_class = tf.squeeze(tf.reshape(label_n_class, [N, H, W, 1]))

    return label_n_class

def mse_4vol(dim=1, batch_size=16):
    self_dim = dim
    def loss_mse_4vol(y_true, y_pred):
        y_true_f1hot = convert_from_1hot_tf(y_true, batch_size=batch_size)
        y_pred_f1hot = convert_from_1hot_tf(y_pred, batch_size=batch_size)
        
        vol_real = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_true_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_pred_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        
        mse_vol = tf.keras.metrics.mean_squared_error(vol_real, vol_pred)
        return mse_vol
    return loss_mse_4vol

def categorical_focal_loss_with_mse_4vol(alpha, batch_size=16, height=256, width=256, gamma=2., dim=1, beta=1):
    alpha = np.array(alpha, dtype=np.float32)
    self_dim = dim
    self_beta = beta
    num_class = len(alpha)

    def cat_focal_loss_with_mse_4vol(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred_clipped = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred_clipped)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred_clipped, gamma) * cross_entropy
        
        # MSE for Volume
        y_true_f1hot = convert_from_1hot_tf(y_true, batch_size=batch_size, num_class=num_class, height=height, width=width)
        y_pred_f1hot = convert_from_1hot_tf(y_pred, batch_size=batch_size, num_class=num_class, height=height, width=width)
        
        vol_real = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_true_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_pred_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        
        mse_vol = tf.keras.metrics.mean_squared_error(vol_real, vol_pred)

        # Compute mean loss in mini_batch
        return backend.mean(backend.sum(loss, axis=-1)) + tf.cast(self_beta * mse_vol, tf.float32)

    return cat_focal_loss_with_mse_4vol

def categorical_focal_loss_wStroke_mse_4vol(alpha, batch_size=16, height=256, width=256, gamma=2., dim=1, beta=1):
    alpha = np.array(alpha, dtype=np.float32)
    self_dim = dim
    self_beta = beta
    num_class = len(alpha)

    def cat_focal_loss_wStroke_mse_4vol(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred_clipped = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred_clipped)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred_clipped, gamma) * cross_entropy
        
        # MSE for Volume
        true_wmh_1hot_tf = tf.gather(y_true, [0,1,2,3], axis=-1)
        pred_wmh_1hot_tf = tf.gather(y_pred, [0,1,2,3], axis=-1)
        y_true_f1hot = convert_from_1hot_tf(true_wmh_1hot_tf, batch_size=batch_size, num_class=4, height=height, width=width)
        y_pred_f1hot = convert_from_1hot_tf(pred_wmh_1hot_tf, batch_size=batch_size, num_class=4, height=height, width=width)
        vol_real = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_true_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_pred_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        
        mse_vol = tf.keras.metrics.mean_squared_error(vol_real, vol_pred)
        
        # Compute mean loss in mini_batch
        return backend.mean(backend.sum(loss, axis=-1)) + tf.cast(self_beta * mse_vol, tf.float32)

    return cat_focal_loss_wStroke_mse_4vol

def categorical_focal_loss_with_mse_4vol_ooi(alpha, batch_size=16, height=256, width=256, gamma=2., dim=1, betha=1, delta=1, theta=1, bce=1):
    alpha = np.array(alpha, dtype=np.float32)
    self_gamma = gamma
    self_dim = dim
    self_bce = bce
    self_betha = betha
    self_delta = delta
    num_class = len(alpha)
    if theta > 0:
        self_theta = 1
    elif theta == 0:
        self_theta = 0
    
    def cat_focal_loss_with_mse_4vol_ooi(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred_clipped = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred_clipped)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred_clipped, gamma) * cross_entropy
        
        # MSE for Volume
        y_true_f1hot = convert_from_1hot_tf(y_true, batch_size=batch_size, num_class=num_class, height=height, width=width)
        y_pred_f1hot = convert_from_1hot_tf(y_pred, batch_size=batch_size, num_class=num_class, height=height, width=width)
        
        vol_real = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_true_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_pred_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        
        mse_vol = tf.keras.metrics.mean_squared_error(vol_real, vol_pred)
        
        # MinVE
        minve_pred_1hot_tf = tf.gather(y_pred, [0,1,3], axis=-1)
        minve_pred_f1hot_tf = convert_from_1hot_tf(minve_pred_1hot_tf, batch_size=batch_size, num_class=3, height=height, width=width)
        minve_vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.equal(minve_pred_f1hot_tf, 2), axis=1), axis=1), 1000), self_dim)
        
        # MaxVE
        maxve_pred_1hot_tf = tf.gather(y_pred, [0,2,3], axis=-1)
        maxve_pred_f1hot_tf = convert_from_1hot_tf(maxve_pred_1hot_tf, batch_size=batch_size, num_class=3, height=height, width=width)
        maxve_vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(maxve_pred_f1hot_tf, axis=1), axis=1), 1000), self_dim)
        
        # IS Out of Interval
        larger_than_maxve = tf.math.greater(vol_real, maxve_vol_pred)
        smaller_than_minve = tf.math.less(vol_real, minve_vol_pred)
        out_of_interval = (tf.cast(tf.math.logical_or(larger_than_maxve, smaller_than_minve), tf.float64) * self_delta) + self_theta
        mse_vol_ooi = tf.math.reduce_mean(tf.math.multiply((vol_real - vol_pred), out_of_interval) ** 2)
        
        ooi = tf.cast(tf.math.logical_or(larger_than_maxve, smaller_than_minve), tf.float64)
        bce_vol_ooi = tf.keras.metrics.binary_crossentropy(tf.zeros((out_of_interval.shape[0],), tf.float32), tf.cast(ooi, tf.float32))
        
        return backend.mean(backend.sum(loss, axis=-1)) + tf.cast(self_betha * mse_vol_ooi, tf.float32) + tf.cast(self_bce * bce_vol_ooi, tf.float32) 
    return cat_focal_loss_with_mse_4vol_ooi


def categorical_focal_loss_wStroke_mse_4vol_ooi(alpha, batch_size=16, height=256, width=256, gamma=2., dim=1, betha=1, delta=1, theta=1, bce=1):
    alpha = np.array(alpha, dtype=np.float32)
    self_gamma = gamma
    self_dim = dim
    self_bce = bce
    self_betha = betha
    self_delta = delta
    num_class = len(alpha)
    if theta > 0:
        self_theta = 1
    elif theta == 0:
        self_theta = 0
    
    def cat_focal_loss_wStroke_mse_4vol_ooi(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred_clipped = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred_clipped)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred_clipped, gamma) * cross_entropy
        
        # MSE for Volume
        true_wmh_1hot_tf = tf.gather(y_true, [0,1,2,3], axis=-1)
        pred_wmh_1hot_tf = tf.gather(y_pred, [0,1,2,3], axis=-1)
        y_true_f1hot = convert_from_1hot_tf(true_wmh_1hot_tf, batch_size=batch_size, num_class=4, height=height, width=width)
        y_pred_f1hot = convert_from_1hot_tf(pred_wmh_1hot_tf, batch_size=batch_size, num_class=4, height=height, width=width)
        vol_real = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_true_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.greater_equal(y_pred_f1hot, 2), axis=1), axis=1), 1000), self_dim)
        
        mse_vol = tf.keras.metrics.mean_squared_error(vol_real, vol_pred)
        
        # MinVE
        minve_pred_1hot_tf = tf.gather(y_pred, [0,1,3], axis=-1)
        minve_pred_f1hot_tf = convert_from_1hot_tf(minve_pred_1hot_tf, batch_size=batch_size, num_class=3, height=height, width=width)
        minve_vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(tf.math.equal(minve_pred_f1hot_tf, 2), axis=1), axis=1), 1000), self_dim)
        
        # MaxVE
        maxve_pred_1hot_tf = tf.gather(y_pred, [0,2,3], axis=-1)
        maxve_pred_f1hot_tf = convert_from_1hot_tf(maxve_pred_1hot_tf, batch_size=batch_size, num_class=3, height=height, width=width)
        maxve_vol_pred = tf.math.multiply(tf.math.divide(tf.math.reduce_sum(tf.math.count_nonzero(maxve_pred_f1hot_tf, axis=1), axis=1), 1000), self_dim)
        
        # IS Out of Interval
        larger_than_maxve = tf.math.greater(vol_real, maxve_vol_pred)
        smaller_than_minve = tf.math.less(vol_real, minve_vol_pred)
        out_of_interval = (tf.cast(tf.math.logical_or(larger_than_maxve, smaller_than_minve), tf.float64) * self_delta) + self_theta
        mse_vol_ooi = tf.math.reduce_mean(tf.math.multiply((vol_real - vol_pred), out_of_interval) ** 2)
        
        ooi = tf.cast(tf.math.logical_or(larger_than_maxve, smaller_than_minve), tf.float64)
        bce_vol_ooi = tf.keras.metrics.binary_crossentropy(tf.zeros((out_of_interval.shape[0],), tf.float32), tf.cast(ooi, tf.float32))
        
        return backend.mean(backend.sum(loss, axis=-1)) + tf.cast(self_betha * mse_vol_ooi, tf.float32) + tf.cast(self_bce * bce_vol_ooi, tf.float32) 
    return cat_focal_loss_wStroke_mse_4vol_ooi