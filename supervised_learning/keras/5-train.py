#!/usr/bin/env python3
"""
Module to train a neural network model using Keras with validation data.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and analyzes
    validation data.

    Args:
        network: The model to train.
        data: A numpy.ndarray of shape (m, nx) containing the input data.
        labels: A one-hot numpy.ndarray of shape (m, classes) containing
                the labels of data.
        batch_size: The size of the batch used for mini-batch gradient descent.
        epochs: The number of passes through data for mini-batch
                gradient descent.
        validation_data: The data to validate the model with, if not None.
        verbose: A boolean that determines if output should be printed
                 during training.
        shuffle: A boolean that determines whether to shuffle the batches
                 every epoch.

    Returns:
        The History object generated after training the model.
    """
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
