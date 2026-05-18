#!/usr/bin/env python3
"""Module that calculates a correlation matrix."""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Args:
        C (numpy.ndarray): Covariance matrix of shape (d, d).
            d is the number of dimensions.

    Returns:
        numpy.ndarray: Correlation matrix of shape (d, d).

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C does not have shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    variance = np.diag(C)
    std_dev = np.sqrt(variance)

    std_outer = np.outer(std_dev, std_dev)

    correlation_matrix = C / std_outer

    return correlation_matrix
