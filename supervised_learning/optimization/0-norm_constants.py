#!/usr/bin/env python3
"""
Calculates the normalization (standardization) constants of a matrix.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in a matrix.
    
    Args:
        X (numpy.ndarray): The matrix of shape (m, nx) to normalize.
            - m is the number of data points.
            - nx is the number of features.
            
    Returns:
        tuple: (mean, std) of each feature, respectively.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    return mean, std
