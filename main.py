# main.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

from datasets.matching_dataset import MatchingDataset
from loggers.logger import Logger
from models.dizygotic_net import DizygoticNet
from trainers.matching_trainer import MatchingTrainer
from utils.config import process_config

import tensorflow as tf


def main(argv) -> None:
    del argv

    # Process the configuration from flags.
    config = process_config()

    if config.mode != "evaluate":
        # Define the datasets.
        train_dataset = MatchingDataset(batch_size=config.batch_size,
                                        folder="datasets/aracati/train",
                                        x_shape=config.input_shape,
                                        y_shape=config.output_shape)

        valid_dataset = MatchingDataset(batch_size=config.batch_size,
                                        folder="datasets/aracati/validation",
                                        x_shape=config.input_shape,
                                        y_shape=config.output_shape)

        # Define the model.
        loss = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        model = DizygoticNet(filters=config.filters, loss=loss, optimizer=optimizer)
        if config.mode == "restore":
            model.load_checkpoint()

        # Define the logger.
        logger = Logger()

        # Define the trainer.
        trainer = MatchingTrainer(model=model, logger=logger, train_dataset=train_dataset, valid_dataset=valid_dataset)
        trainer.train()
    else:
        logging.fatal("Evaluation mode is not yet implemented.")
        exit(1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
