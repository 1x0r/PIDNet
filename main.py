import tensorflow as tf
from pidnet.pidnet import PIDNet

if __name__ == '__main__':

    seg_model = PIDNet(m=2, n=3, num_classes=11, planes=64, ppm_planes=96, head_planes=128, augment=True)
    seg_model.build(input_shape=(8, 480, 480, 3))
    seg_model.summary()
