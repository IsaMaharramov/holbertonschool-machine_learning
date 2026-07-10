#!/usr/bin/env python3
"""
Module to convert a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: A numpy array (or array-like) containing the labels to convert.
        classes (int): The total number of classes.

    Returns:
        The one-hot matrix.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
