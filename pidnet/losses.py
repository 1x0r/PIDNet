import tensorflow as tf


class OHEMCrossentropy(tf.keras.losses.Loss):

    def __init__(self, threshold=0.7, min_kept=100000, balance_weights=(0.5, 0.5), weight=None):
        super(OHEMCrossentropy, self).__init__()

        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.balance_weights = balance_weights

    def _ce_call(self, y_true, y_pred):

        pixel_losses = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True, axis=-1).reshape(-1)

        return tf.reduce_mean(pixel_losses)
    
    def _ohem_call(self, y_true, y_pred):

        pixel_losses = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True, axis=-1).reshape(-1)
        pixel_losses = tf.sort(pixel_losses, direction='DESCENDING')[:min(self.min_kept, pixel_losses.shape[0] - 1)]
        pixel_losses = pixel_losses[pixel_losses < self.threshold]

        return tf.reduce_mean(pixel_losses)

    def call(self, y_true, y_pred):

        if not (isinstance(y_pred, list) or isinstance(y_pred, tuple)):
            y_pred = [y_pred]

        if len(self.balance_weights) == len(y_pred):
            losses = [self._ce_call] * (len(y_pred) - 1) + [self._ohem_call]
            return sum([w * f(x, y_true) for w, x, f in zip(self.balance_weights, y_pred, losses) ])
        else:
            raise ValueError("lengths of prediction and target are not identical!")


class BoundaryLoss(tf.keras.losses.Loss):

    def __init__(self, coeff_bce=20.0):
        super(BoundaryLoss, self).__init__()

        self.coeff_bce = coeff_bce

    def _weighted_bce(self, y_true, y_pred):
        y_true_ravel = tf.reshape(y_true, (1, -1))
        y_pred_ravel = tf.reshape(y_pred, (1, -1))

        pos_index = (y_true_ravel == 1)
        neg_index = (y_true_ravel == 0)
        pos_num = tf.reduce_sum(pos_index)
        neg_num = tf.reduce_sum(neg_index)
        sum_num = pos_num + neg_num

        weights = tf.zeros_like(y_pred_ravel)
        weights[pos_index] = pos_num * 1.0 / sum_num
        weights[neg_index] = neg_num * 1.0 / sum_num

        return tf.reduce_mean(tf.math.multiply(weights, tf.keras.losses.binary_crossentropy(y_true_ravel, y_pred_ravel, from_logits=True, axis=-1)))

    def call(self, y_true, y_pred):

        bce_loss = self.coeff_bce * self._weighted_bce(y_true, y_pred)

        return bce_loss