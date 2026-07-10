#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum optimization algorithm.
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.
    
    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v (numpy.ndarray): The previous first moment of var.
        
    Returns:
        tuple: The updated variable and the new moment, respectively.
    """
    # Calculate the new moment
    v_new = beta1 * v + (1 - beta1) * grad
    
    # Update the variable
    var_new = var - alpha * v_new
    
    return var_new, v_new
