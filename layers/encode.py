# encode.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, Union

import tensorflow as tf


class Encode(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]],
                 activation_fn: type(tf.keras.layers.Layer), with_pool=True) -> None:
        super(Encode, self).__init__(trainable=True, name=None)

        # Save parameters to class.
        self.activation_fn = activation_fn
        self.filters = filters
        self.kernel_size = kernel_size
        self.with_pool = with_pool

        # Create encode sublayers.
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.activation_1 = activation_fn()
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.activation_2 = activation_fn()

        if with_pool:
            self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

    def call(self, inputs, training=False) -> tf.Tensor:
        x = self.conv_1(inputs)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.activation_2(x)

        if self.with_pool:
            x = self.max_pool(x)

        return x
