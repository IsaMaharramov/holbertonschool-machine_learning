#!/usr/bin/env python3
"""
Cost Module for t-SNE
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation using
    Kullback-Leibler (KL) divergence.

    Args:
        P (numpy.ndarray): Array of shape (n, n) with the P affinities.
        Q (numpy.ndarray): Array of shape (n, n) with the Q affinities.

    Returns:
        float: C, the cost of the transformation.
    """
    # Prevent division by zero and log(0) errors by applying a lower bound
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)

    # Calculate KL divergence: sum(P * log(P / Q))
    # Where P is exactly 0, cost contribution should logically be 0, which
    # multiplying by the actual P matrix handles automatically.
    C = np.sum(P * np.log(P_safe / Q_safe))

    return C
