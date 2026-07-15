#!/usr/bin/env python3
"""
Module that updates the weights and biases of a neural network
using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.
    
    Args:
        Y: One-hot numpy.ndarray of shape (classes, m) that contains
           the correct labels for the data.
        weights: Dictionary of the weights and biases of the neural network.
        cache: Dictionary of the outputs of each layer of the neural network.
        alpha: The learning rate.
        lambtha: The L2 regularization parameter.
        L: The number of layers of the network.
    """
    m = Y.shape[1]
    
    # Calculate initial dz for the output layer (Softmax)
    dz = cache['A' + str(L)] - Y
    
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        
        # Calculate gradients with L2 regularization penalty
        dw = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        
        # Calculate dz for the next iteration (previous layer)
        if i > 1:
            da = np.matmul(W.T, dz)
            # Derivative of tanh activation function
            dz = da * (1 - (A_prev ** 2))
            
        # Update weights and biases in place
        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db
