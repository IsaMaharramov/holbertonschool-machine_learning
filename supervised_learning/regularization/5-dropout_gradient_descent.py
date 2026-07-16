#!/usr/bin/env python3
"""
Module that updates the weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: One-hot numpy.ndarray of shape (classes, m) with correct labels.
        weights: Dictionary of the weights and biases.
        cache: Dictionary of the outputs and dropout masks of each layer.
        alpha: The learning rate.
        keep_prob: The probability that a node will be kept.
        L: The number of layers of the network.
    """
    m = Y.shape[1]

    # Calculate initial dz for the output layer (Softmax)
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Gradients for current layer
        dw = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # Calculate dz for the next iteration (previous layer)
        if i > 1:
            da = np.matmul(W.T, dz)
            # Apply the dropout mask from the forward pass
            D = cache['D' + str(i - 1)]
            da = (da * D) / keep_prob
            # Derivative of tanh activation function
            dz = da * (1 - (A_prev ** 2))

        # Update weights and biases in place
        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db
