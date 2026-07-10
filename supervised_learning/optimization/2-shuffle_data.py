#!/usr/bin/env python3
"""
Shuffles the data points in two matrices the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles two matrices synchronously.
    
    Args:
        X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle.
            - m is the number of data points.
            - nx is the number of features in X.
        Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle.
            - m is the same number of data points as in X.
            - ny is the number of features in Y.
            
    Returns:
        tuple: The shuffled X and Y matrices.
    """
    # Generate a random permutation of indices from 0 to m - 1
    permutation = np.random.permutation(X.shape[0])
    
    # Apply the same permutation to both X and Y
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    
    return X_shuffled, Y_shuffled
