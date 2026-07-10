#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix based on given mean and standard deviation.
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes a matrix.

    Args:
        X (numpy.ndarray): The matrix of shape (d, nx) to normalize.
            - d is the number of data points.
            - nx is the number of features.
        m (numpy.ndarray): Array of shape (nx,) containing the mean of
                           all features of X.
        s (numpy.ndarray): Array of shape (nx,) containing the standard
                           deviation of all features of X.

    Returns:
        numpy.ndarray: The normalized X matrix.
    """
    return (X - m) / s
