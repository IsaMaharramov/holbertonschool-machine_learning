#!/usr/bin/env python3
"""Module that calculates the mean and covariance of a data set."""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.
    
    Args:
        X (numpy.ndarray): Array of shape (n, d) containing the data set.
            n is the number of data points.
            d is the number of dimensions in each data point.

    Returns:
        tuple: (mean, cov)
            mean is a numpy.ndarray of shape (1, d) containing the mean.
            cov is a numpy.ndarray of shape (d, d) containing the covariance.
            
    Raises:
        TypeError: If X is not a 2D numpy.ndarray.
        ValueError: If n is less than 2.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
