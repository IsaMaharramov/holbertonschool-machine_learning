#!/usr/bin/env python3
"""
Module to save and load a Keras model's weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file.

    Args:
        network: The model whose weights should be saved.
        filename (str): The path of the file that the weights
                        should be saved to.
        save_format (str): The format in which the weights should be saved.

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights from a file.

    Args:
        network: The model to which the weights should be loaded.
        filename (str): The path of the file that the weights
                        should be loaded from.

    Returns:
        None
    """
    network.load_weights(filename)
    return None
