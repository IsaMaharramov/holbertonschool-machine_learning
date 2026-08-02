#!/usr/bin/env python3
"""
PCA v2 Module for Dimensionality Reduction
"""
import numpy as np


def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): A dataset of shape (n, d) where:
            - n is the number of data points.
            - d is the number of dimensions in each point.
        ndim (int): The new dimensionality of the transformed X.

    Returns:
        numpy.ndarray: T, an array of shape (n, ndim) containing the
            transformed version of X.
    """
    # Mean-center the data (subtract the mean of each dimension)
    X_centered = X - np.mean(X, axis=0)

    # Perform Singular Value Decomposition (SVD) on the centered data
    U, S, Vh = np.linalg.svd(X_centered)

    # Extract the top `ndim` principal components (columns of Vh.T)
    W = Vh.T[:, :ndim]

    # Project the centered data onto the principal components
    T = np.matmul(X_centered, W)

    return T
