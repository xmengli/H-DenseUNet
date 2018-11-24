import keras.backend as K
import tensorflow as tf


def weighted_crossentropy(y_true, y_pred):
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    y_pred_f = K.reshape(y_pred, (-1,3))
    y_true_f = K.reshape(y_true, (-1,))

    soft_pred_f = K.softmax(y_pred_f)
    soft_pred_f = K.log(tf.clip_by_value(soft_pred_f, 1e-10, 1.0))

    neg = K.equal(y_true_f, K.zeros_like(y_true_f))
    neg_calculoss = tf.gather(soft_pred_f[:,0], tf.where(neg))

    pos1 = K.equal(y_true_f, K.ones_like(y_true_f))
    pos1_calculoss = tf.gather(soft_pred_f[:,1], tf.where(pos1))

    pos2 = K.equal(y_true_f, 2*K.ones_like(y_true_f))
    pos2_calculoss = tf.gather(soft_pred_f[:,2], tf.where(pos2))

    loss = -K.mean(tf.concat([0.78*neg_calculoss, 0.65*pos1_calculoss, 8.57*pos2_calculoss], 0))

    return loss

def weighted_crossentropy_2ddense(y_true, y_pred):

    y_pred_f = K.reshape(y_pred, (-1,3))
    y_true_f = K.reshape(y_true, (-1,))

    soft_pred_f = K.softmax(y_pred_f)
    soft_pred_f = K.log(tf.clip_by_value(soft_pred_f, 1e-10, 1.0))

    neg = K.equal(y_true_f, K.zeros_like(y_true_f))
    neg_calculoss = tf.gather(soft_pred_f[:,0], tf.where(neg))

    pos1 = K.equal(y_true_f, K.ones_like(y_true_f))
    pos1_calculoss = tf.gather(soft_pred_f[:,1], tf.where(pos1))

    pos2 = K.equal(y_true_f, 2*K.ones_like(y_true_f))
    pos2_calculoss = tf.gather(soft_pred_f[:,2], tf.where(pos2))

    loss = -K.mean(tf.concat([0.78*neg_calculoss, 0.65*pos1_calculoss, 8.57*pos2_calculoss], 0))

    return loss

