#!/usr/bin/env python3
"""
Entropy Module for t-SNE
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point.

    Args:
        Di (numpy.ndarray): Array of shape (n - 1,) containing pairwise 
            distances between a data point and all other points.
        beta (numpy.ndarray): Array of shape (1,) containing the beta value.

    Returns:
        tuple: (Hi, Pi)
            - Hi: The Shannon entropy of the points.
            - Pi: numpy.ndarray of shape (n - 1,) with P affinities.
    """
    # Numerator of P affinities calculation
    P_num = np.exp(-Di * beta)
    
    # Denominator (sum of numerators)
    sum_P = np.sum(P_num)
    
    # Calculate the normalized P affinities
    Pi = P_num / sum_P
    
    # Calculate Shannon Entropy using the explicit definition
    # np.maximum prevents -inf errors in log2(0)
    Pi_safe = np.maximum(Pi, 1e-12)
    Hi = -np.sum(Pi * np.log2(Pi_safe))

    return Hi, Pi
