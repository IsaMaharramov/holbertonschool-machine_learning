#!/usr/bin/env python3
"""
Initialize t-SNE Module
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) to be transformed.
        perplexity (float): The perplexity for all Gaussian distributions.

    Returns:
        tuple: (D, P, betas, H)
            - D: numpy.ndarray of shape (n, n) with pairwise squared distances.
            - P: numpy.ndarray of shape (n, n) initialized to all 0s.
            - betas: numpy.ndarray of shape (n, 1) initialized to all 1s.
            - H: The Shannon entropy for perplexity with a base of 2.
    """
    n, d = X.shape

    # Calculate squared pairwise distances using vectorized expansion:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
    sum_X = np.sum(np.square(X), axis=1)
    D = -2 * np.matmul(X, X.T) + sum_X + sum_X.reshape(-1, 1)

    # Ensure the diagonal is exactly 0 and prevent negative float inaccuracies
    np.fill_diagonal(D, 0)
    D = np.maximum(D, 0)

    # Initialize P affinities to 0 and betas to 1
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    # Calculate Shannon entropy from perplexity
    H = np.log2(perplexity)

    return D, P, betas, H
