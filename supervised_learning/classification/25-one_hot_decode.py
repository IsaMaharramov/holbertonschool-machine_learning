#!/usr/bin/env python3
"""
Module that contains the function one_hot_decode.
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): shape (classes, m) encoded matrix.

    Returns:
        A numpy.ndarray with shape (m,) containing numeric labels,
        or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # argmax along axis 0 finds the row index of the maximum value (the 1)
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
