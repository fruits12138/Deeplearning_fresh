import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# 13layers

conv_layers = [  # 5 units of conv+max pooling
    # unit1
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit2
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit3
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit4
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit5
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),  # 为了使得参数的size完全相同
    layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
]


def main():
    # b[b,32,32,3]->[b,1,1,512]
    conv_net = Sequential([conv_layers])
    conv_net.build(input_shape=[None, 32, 32, 3])
    # x = tf.random.normal([4, 32, 32, 3])
    # out = conv_net(x)
    # print(out.shape)

    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),
    ])

    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[])


if __name__ == '__main__':
    main()
