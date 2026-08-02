#!/usr/bin/env python3
"""
Gradients Module for t-SNE
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): Array of shape (n, ndim) containing the low
            dimensional transformation of X.
        P (numpy.ndarray): Array of shape (n, n) containing the P affinities.

    Returns:
        tuple: (dY, Q)
            - dY: numpy.ndarray of shape (n, ndim) containing the gradients.
            - Q: numpy.ndarray of shape (n, n) containing the Q affinities.
    """
    # Get Q affinities and the numerator using the external function
    Q, num = Q_affinities(Y)

    # Calculate the gradient difference multiplier: (P - Q) * num
    PQ_diff = (P - Q) * num

    # Vectorized gradient calculation:
    # dY = sum((p_ij - q_ij) * num_ij * (y_i - y_j))
    # We can separate this into: y_i * sum(...) - sum(... * y_j)
    dY = np.sum(PQ_diff, axis=1, keepdims=True) * Y - np.matmul(PQ_diff, Y)

    return dY, Q
