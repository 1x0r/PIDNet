import tensorflow as tf


class FullModel(tf.keras.models.Model):

    def __init__(self, model, sem_loss, bd_loss):
        super(FullModel, self).__init__()

        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
    
    def call(self, inputs):
        return self.model(inputs)

    def compute_loss(self, x, y, y_pred, sample_weight):
        """
        y: y[0] - image, y[1] - edges
        """
        outputs = self.model(x)

        h, w = y[0].shape[1], y[0].shape[2]
        ph, pw = outputs[0].shape[1], outputs[0].shape[2]

        if ph != h or pw != w:
            for i in range(len(outputs)):
                outputs[i] = tf.image.resize(outputs[i], size=(h, w))

        loss_s = self.sem_loss(y[0], outputs[:-1])
        loss_b = self.bd_loss(y[1], outputs[-1])

        filler = -tf.ones_like(y)
        bd_label = tf.where(tf.sigmoid(outputs[-1][..., 0] > 0.8), y, filler)
        loss_sb = self.sem_loss(outputs[-2], bd_label)

        return loss_s + loss_b + loss_sb

