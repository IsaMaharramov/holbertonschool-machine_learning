#!/usr/bin/env python3
"""
Module to save and load a Keras model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Args:
        network: The model to save.
        filename (str): The path of the file that the model should be saved to.

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Args:
        filename (str): The path of the file that the model should
                        be loaded from.

    Returns:
        The loaded Keras model.
    """
    return K.models.load_model(filename)
