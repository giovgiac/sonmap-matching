# base_model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import abc
import os
import tensorflow as tf


class BaseModel(abc.ABC, tf.keras.Model):
    def __init__(self, loss: tf.keras.losses.Loss, optimizer: tf.keras.optimizers.Optimizer, name: str) -> None:
        super(BaseModel, self).__init__(name=name)

        # Initialize configuration and state variables.
        self.config = flags.FLAGS
        self.epoch = tf.Variable(initial_value=0, shape=(), dtype=tf.int32, trainable=False, name="epoch")
        self.global_step = tf.Variable(initial_value=0, shape=(), dtype=tf.int64, trainable=False, name="global_step")
        self.loss = loss
        self.optimizer = optimizer
        self.saver = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        raise NotImplementedError

    def load_checkpoint(self) -> None:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=self.config.checkpoint_dir)
        if latest_checkpoint:
            logging.info("Restoring Model Checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(save_path=latest_checkpoint)
            logging.info("[Epoch {}] Model Restored...".format(int(self.epoch)))

    def save_checkpoint(self) -> None:
        logging.info("[Epoch {}] Saving Model...".format(int(self.epoch)))
        self.saver.save(file_prefix=os.path.join(self.config.checkpoint_dir, "ckpt"))
        logging.info("[Epoch {}] Model Saved...".format(int(self.epoch)))
