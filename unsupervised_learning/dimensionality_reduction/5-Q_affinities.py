#!/usr/bin/env python3
"""
Q Affinities Module for t-SNE
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities of the low dimensional data.

    Args:
        Y (numpy.ndarray): Array of shape (n, ndim) containing the low
            dimensional transformation of X.

    Returns:
        tuple: (Q, num)
            - Q: numpy.ndarray of shape (n, n) containing the Q affinities.
            - num: numpy.ndarray of shape (n, n) containing the numerator
                   of the Q affinities.
    """
    # Calculate pairwise squared distances of Y
    sum_Y = np.sum(np.square(Y), axis=1)
    D_Y = sum_Y + sum_Y.reshape(-1, 1) - 2 * np.matmul(Y, Y.T)

    # Avoid negative distances due to floating-point errors
    D_Y = np.maximum(D_Y, 0)

    # Numerator calculation: (1 + ||yi - yj||^2)^-1
    num = 1 / (1 + D_Y)
    np.fill_diagonal(num, 0)

    # Q affinities calculation: normalize the numerator
    Q = num / np.sum(num)

    return Q, num
