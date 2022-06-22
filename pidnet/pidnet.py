import tensorflow as tf

from .utils import BasicBlock, Bottleneck, SegmentHead, DAPPM, PAPPM, PagFM, Bag, LightBag
from .losses import OHEMCrossentropy, BoundaryLoss

bn_momentum = 0.1


class PIDNet(tf.keras.models.Model):

    def __init__(self, m=2, n=3, num_classes=11, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()

        self.augment = augment

        # I-branch

        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform'),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
            tf.keras.layers.ReLU()
        ], name='pid_net_conv1')

        self.relu = tf.keras.layers.ReLU(name='pidnet_relu')

        self.layer1_i = self._make_layer(BasicBlock, planes, planes, num_blocks=m, stride=1)
        self.layer2_i = self._make_layer(BasicBlock, planes, planes * 2, num_blocks=m, stride=2)
        self.layer3_i = self._make_layer(BasicBlock, planes * 2, planes * 4, num_blocks=n, stride=2)
        self.layer4_i = self._make_layer(BasicBlock, planes * 4, planes * 8, num_blocks=n, stride=2)
        self.layer5_i = self._make_layer(Bottleneck, planes * 8, planes * 8, num_blocks=2, stride=2)

        # P-branch

        self.compression3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
        ], name='pidnet_compression3')

        self.compression4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=1, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum),
        ], name='pidnet_compression4')

        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, num_blocks=m, stride=1)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, num_blocks=m, stride=1)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, num_blocks=1, stride=1)

        # D-branch

        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes, stride=1)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, num_blocks=1, stride=1)

            self.diff3 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff3')

            self.diff4 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff4')

            self.spp = PAPPM(ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4)

        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2, stride=1)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2, stride=1)
            
            self.diff3 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff3')

            self.diff4 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * 2, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ], name='pidnet_diff4')

            self.spp = DAPPM(ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, num_blocks=1, stride=1)

        # Prediction Head

        if self.augment:
            self.seghead_p = SegmentHead(head_planes, num_classes)
            self.seghead_d = SegmentHead(planes, 1)

        self.final_layer = SegmentHead(head_planes, num_classes)


    def call(self, inputs, training=False):

        width_output = tf.shape(inputs)[2] // 8
        height_output = tf.shape(inputs)[1] // 8

        # I-stream

        x = self.conv1(inputs)
        x = self.layer1_i(x)
        x = self.relu(self.layer2_i(self.relu(x)))

        x_p = self.layer3_p(x) # p-branch
        x_d = self.layer3_d(x) # d-branch

        x = self.relu(self.layer3_i(x))
        x_p = self.pag3([x_p, self.compression3(x)])
        x_d += tf.image.resize(self.diff3(x), size=(height_output, width_output))

        if self.augment:
            temp_p = x_p

        x = self.relu(self.layer4_i(x))
        x_p = self.layer4_p(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))

        x_p = self.pag4([x_p, self.compression4(x)])
        x_d += tf.image.resize(self.diff4(x), size=(height_output, width_output))

        if self.augment:
            temp_d = x_d

        x_p = self.layer5_p(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))

        x = tf.image.resize(self.spp(self.layer5_i(x)), size=(height_output, width_output))
        x_dfm = self.dfm([x_p, x, x_d])

        x_p = self.final_layer(x_dfm)

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_p, x_extra_d]
        else:
            return x_p

    def _make_single_layer(self, block, inplanes, planes, stride=1):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ])

        layer = block(planes, stride=stride, downsample=downsample, no_relu=True)

        return layer

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):

        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer='he_uniform'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum)
            ])

        no_relu = False
        layers = []
        layers.append(block(planes, stride=stride, downsample=downsample, no_relu=no_relu))
        for i in range(1, num_blocks):
            if i == num_blocks - 1:
                no_relu = True
            layers.append(block(planes, stride=1, no_relu=no_relu))

        return tf.keras.Sequential(layers)