#!/usr/bin/env python3
"""
Module that contains the function one_hot_encode.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): shape (m,) containing numeric class labels.
        classes (int): maximum number of classes found in Y.

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if len(Y) == 0 or classes <= np.max(Y):
        return None

    # Create an identity matrix of size (classes, classes)
    # Use Y as indices to pick the corresponding rows
    # Transpose to get shape (classes, m)
    one_hot = np.eye(classes)[:, Y]

    return one_hot
