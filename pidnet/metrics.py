import tensorflow as tf


def pixel_acc(pred, label):
    preds = tf.reduce_max(pred, axis=-1)
    valid = tf.cast(label >= 0, tf.int32)
    acc_sum = tf.reduce_sum(valid * tf.cast(preds == label, tf.int32))
    pixel_sum = tf.reduce_sum(valid)
    acc = tf.cast(acc_sum, tf.float32) / (tf.cast(pixel_sum, tf.float32) + 1e-10)
    return acc