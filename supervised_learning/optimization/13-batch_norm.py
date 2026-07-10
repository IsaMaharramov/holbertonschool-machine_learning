#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network.
    
    Args:
        Z (numpy.ndarray): Matrix of shape (m, n) to normalize.
        gamma (numpy.ndarray): Matrix of shape (1, n) containing the scales.
        beta (numpy.ndarray): Matrix of shape (1, n) containing the offsets.
        epsilon (float): A small number used to avoid division by zero.
        
    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    
    # Normalize the data
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    
    # Scale and shift
    Z_out = gamma * Z_norm + beta
    
    return Z_out
