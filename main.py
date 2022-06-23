import tensorflow as tf
from pidnet.pidnet import PIDNet
from pidnet.losses import OHEMCrossentropy, BoundaryLoss
from pidnet.model import FullModel

import numpy as np

if __name__ == '__main__':

    seg_model = PIDNet((480, 480, 3), m=2, n=3, num_classes=11, planes=64, ppm_planes=96, head_planes=128, augment=True)
    seg_model.compile(loss='mse')
    seg_model.fit(x=np.random.uniform(size=(2, 480, 480, 3)), y=np.zeros((2, 60, 60, 11)))



    # ohem_loss = OHEMCrossentropy(threshold=0.9, min_kept=100000, balance_weights=(0.5, 0.5, 0.5))
    # boundary_loss = BoundaryLoss()

    # full_model = FullModel(seg_model, ohem_loss, boundary_loss)
    # full_model.build(input_shape=(None, 480, 480, 3))
    # full_model.summary()

