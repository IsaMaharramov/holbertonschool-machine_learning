#!/usr/bin/env python3
"""
Module to make predictions using a neural network in Keras.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network: The network model to make the prediction with.
        data: The input data to make the prediction with.
        verbose: A boolean that determines if output should be
                 printed during the prediction process.

    Returns:
        The prediction for the data.
    """
    return network.predict(x=data, verbose=verbose)
