#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set with bias correction.
"""


def moving_average(data, beta):
    """
    Calculates the exponentially weighted moving average with bias correction.
    
    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.
        
    Returns:
        list: A list containing the moving averages of data.
    """
    v = 0
    moving_averages = []
    
    for t in range(1, len(data) + 1):
        # Calculate exponentially weighted moving average
        v = beta * v + (1 - beta) * data[t - 1]
        
        # Apply bias correction
        bias_corrected_v = v / (1 - (beta ** t))
        
        moving_averages.append(bias_corrected_v)
        
    return moving_averages
