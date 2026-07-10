#!/usr/bin/env python3
"""
Module to train a neural network model using Keras with validation data,
early stopping, and learning rate decay.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping
    and learning rate decay.

    Args:
        network: The model to train.
        data: A numpy.ndarray of shape (m, nx) containing the input data.
        labels: A one-hot numpy.ndarray of shape (m, classes) containing
                the labels of data.
        batch_size: The size of the batch used for mini-batch gradient descent.
        epochs: The number of passes through data for mini-batch descent.
        validation_data: The data to validate the model with, if not None.
        early_stopping: A boolean that indicates whether early stopping
                        should be used.
        patience: The patience used for early stopping.
        learning_rate_decay: A boolean that indicates whether learning rate
                             decay should be used.
        alpha: The initial learning rate.
        decay_rate: The decay rate.
        verbose: A boolean that determines if output should be printed.
        shuffle: A boolean that determines whether to shuffle the batches
                 every epoch.

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    if validation_data:
        # Early Stopping
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
            callbacks.append(early_stop)

        # Learning Rate Decay (Inverse Time Decay)
        if learning_rate_decay:
            def lr_schedule(epoch):
                """ Stepwise inverse time decay function """
                return alpha / (1 + decay_rate * epoch)

            # verbose=1 prints a message every time the learning rate updates
            lr_decay = K.callbacks.LearningRateScheduler(lr_schedule,
                                                         verbose=1)
            callbacks.append(lr_decay)

    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
