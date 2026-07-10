#!/usr/bin/env python3
"""
Updates a variable using the RMSProp optimization algorithm.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        s (numpy.ndarray): The previous second moment of var.

    Returns:
        tuple: The updated variable and the new moment, respectively.
    """
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    return var_new, s_new
