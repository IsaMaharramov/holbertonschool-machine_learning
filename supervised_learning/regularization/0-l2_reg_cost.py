#!/usr/bin/env python3
"""
Module that calculates the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: The cost of the network without L2 regularization.
        lambtha: The regularization parameter.
        weights: Dictionary of the weights and biases of the neural network.
        L: The number of layers in the neural network.
        m: The number of data points used.

    Returns:
        The cost of the network accounting for L2 regularization.
    """
    l2_penalty = 0
    for i in range(1, L + 1):
        l2_penalty += np.linalg.norm(weights['W' + str(i)]) ** 2

    l2_cost = cost + (lambtha / (2 * m)) * l2_penalty

    return l2_cost
