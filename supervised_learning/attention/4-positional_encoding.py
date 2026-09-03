#!/usr/bin/env python3
"""
Positional Encoding module
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    :param max_seq_len: integer representing maximum sequence length
    :param dm: integer representing model depth (dimensionality)
    :return: numpy.ndarray of shape (max_seq_len, dm) containing
             the positional encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    angle_rads = pos * angle_rates

    # Apply sin to even indices (2i); 2i < dm
    PE[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices (2i+1); 2i+1 < dm
    PE[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return PE
