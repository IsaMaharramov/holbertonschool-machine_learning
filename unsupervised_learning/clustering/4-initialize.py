#!/usr/bin/env python3
"""
Module -> initializing GMM variables.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables -> a Gaussian Mixture Model.

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - k: positive integer containing the number of clusters

    Returns:
    - pi: numpy.ndarray of shape (k,) containing priors, initialized evenly
    - m: numpy.ndarray of shape (k, d) containing centroid means, via K-means
    - S: numpy.ndarray of shape (k, d, d) containing covariance matrices,
         initialized as identity matrices
    - Returns None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None

    # priors -> each cluster, initialized evenly
    pi = np.ones(k) / k

    # centroid means -> each cluster, initialized with K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # covariance matrices -> each cluster, initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    return pi, m, S
