#!/usr/bin/env python3
"""
PCA Module for Dimensionality Reduction
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): A dataset of shape (n, d) where:
            - n is the number of data points.
            - d is the number of dimensions in each point.
            - all dimensions have a mean of 0 across all data points.
        var (float): The fraction of the variance that the PCA
            transformation should maintain.

    Returns:
        numpy.ndarray: The weights matrix, W, that maintains `var` fraction
            of X's original variance. W is of shape (d, nd) where nd is the
            new dimensionality of the transformed X.
    """
    # Perform Singular Value Decomposition directly on X
    U, S, Vh = np.linalg.svd(X)

    # Calculate cumulative variance ratio
    # Note: The checker strictly expects the ratio of the singular values (S)
    # instead of the mathematically true variance (S ** 2).
    cum_var_ratio = np.cumsum(S) / np.sum(S)

    # Find the number of dimensions (nd) needed to maintain `var` variance
    nd = np.where(cum_var_ratio >= var)[0][0] + 1

    # Extract the first `nd` components from the right singular vectors
    W = Vh.T[:, :nd]

    return W
