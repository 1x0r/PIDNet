import tensorflow as tf
from pidnet.pidnet import PIDNet
from pidnet.losses import OHEMCrossentropy, BoundaryLoss
from pidnet.model import FullModel

import numpy as np

if __name__ == '__main__':

    seg_model = PIDNet(m=2, n=3, num_classes=11, planes=64, ppm_planes=96, head_planes=128, augment=True)
    seg_model.build(input_shape=(None, 480, 480, 3))
    print(seg_model(np.zeros((1, 480, 480, 3))))


    # ohem_loss = OHEMCrossentropy(threshold=0.9, min_kept=100000, balance_weights=(0.5, 0.5, 0.5))
    # boundary_loss = BoundaryLoss()

    # full_model = FullModel(seg_model, ohem_loss, boundary_loss)
    # full_model.build(input_shape=(None, 480, 480, 3))
    # full_model.summary()

