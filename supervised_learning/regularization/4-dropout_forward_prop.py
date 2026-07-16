#!/usr/bin/env python3
"""
Module that conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data.
        weights: Dictionary of the weights and biases of the neural network.
        L: The number of layers in the network.
        keep_prob: The probability that a node will be kept.

    Returns:
        A dictionary containing the outputs of each layer and the
        dropout mask used on each layer.
    """
    cache = {}
    cache['A0'] = X
    A_prev = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, A_prev) + b

        if i < L:
            # Tanh activation for hidden layers
            A = np.tanh(Z)
            # Create dropout mask (1s and 0s)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            # Apply mask and scale (Inverted Dropout)
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D
        else:
            # Softmax activation for the final layer
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)

        cache['A' + str(i)] = A
        A_prev = A

    return cache
