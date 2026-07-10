#!/usr/bin/env python3
"""
Updates a variable in place using the Adam optimization algorithm.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.
    
    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight used for the first moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v (numpy.ndarray): The previous first moment of var.
        s (numpy.ndarray): The previous second moment of var.
        t (int): The time step used for bias correction.
        
    Returns:
        tuple: The updated variable, the new first moment, and the 
               new second moment, respectively.
    """
    # First moment update
    v_new = beta1 * v + (1 - beta1) * grad
    
    # Second moment update
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    
    # Bias corrections
    v_corrected = v_new / (1 - (beta1 ** t))
    s_corrected = s_new / (1 - (beta2 ** t))
    
    # Variable update
    var_new = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    
    return var_new, v_new, s_new
