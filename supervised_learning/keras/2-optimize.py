#!/usr/bin/env python3
"""
Module that sets up Adam optimization for a Keras model.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical crossentropy
    loss and accuracy metrics.

    Args:
        network: The model to optimize.
        alpha (float): The learning rate.
        beta1 (float): The first Adam optimization parameter.
        beta2 (float): The second Adam optimization parameter.

    Returns:
        None
    """
    opt = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
