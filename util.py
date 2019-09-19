import numpy as np
from keras import backend as K
import tensorflow as tf
import cv2


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def balance_loss(y_true, y_pred):
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    # balance positive and negative samples in an image
    beta = 1 - tf.reduce_mean(labels)
    # first apply sigmoid activation
    predicts = tf.nn.sigmoid(logits)
    # log +epsilon for stable cal
    loss = tf.reduce_mean(
        -1 * (beta * labels * tf.log(predicts + 1e-4) +
              (1 - beta) * (1 - labels) * tf.log(1 - predicts + 1e-4)))
    return loss


def normalize(img):
    return img / 255


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def weights_loss(y_true, y_pred):
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    w_loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=90.0)
    return w_loss


def canny():
    img = cv2.imread("test/3.jpg", 0)
    canny_im = cv2.Canny(img, 100, 50)
    cv2.imwrite('3.png', canny_im)

canny()