# dizygotic_net.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from base.base_model import BaseModel
from layers.encode import Encode

import tensorflow as tf


class DizygoticNet(BaseModel):
    def __init__(self, filters: int, loss: tf.keras.losses.Loss, optimizer: tf.keras.optimizers.Optimizer):
        super(DizygoticNet, self).__init__(loss, optimizer, name="DizygoticNet")

        # Store network architecture hyperparameters.
        self.filters = filters

        # Define sublayers of the Dizygotic Network.

        # Sonar encoding layers.
        self.son_e1 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.son_e2 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.son_e3 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.son_e4 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.son_e5 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.son_f = tf.keras.layers.Flatten()

        # Satellite encoding layers.
        self.sat_e1 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.sat_e2 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.sat_e3 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.sat_e4 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.sat_e5 = Encode(filters=filters, kernel_size=3, activation_fn=tf.keras.layers.ReLU, with_pool=True)
        self.sat_f = tf.keras.layers.Flatten()

        self.son_sat_cat = tf.keras.layers.Concatenate()

        # Multilayer perceptron to match encodings.
        self.dense_1 = tf.keras.layers.Dense(units=2048, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        [x, y] = inputs

        # Sonar
        son = self.son_e1(x, training=training)
        son = self.son_e2(son, training=training)
        son = self.son_e3(son, training=training)
        son = self.son_e4(son, training=training)
        son = self.son_e5(son, training=training)
        son = self.son_f(son)

        # Satellite
        sat = self.sat_e1(y, training=training)
        sat = self.sat_e2(sat, training=training)
        sat = self.sat_e3(sat, training=training)
        sat = self.sat_e4(sat, training=training)
        sat = self.sat_e5(sat, training=training)
        sat = self.sat_f(sat)

        # Concatenate
        z = self.son_sat_cat([son, sat])

        # MLP
        z = self.dense_1(z)
        z = self.dense_2(z)
        z = self.dense_3(z)

        return self.dense_4(z)
