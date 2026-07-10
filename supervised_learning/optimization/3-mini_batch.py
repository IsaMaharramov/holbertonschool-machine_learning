#!/usr/bin/env python3
"""
Creates mini-batches to be used for training a neural network using
mini-batch gradient descent.
"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for training.
    
    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) representing input data.
        Y (numpy.ndarray): Matrix of shape (m, ny) representing the labels.
        batch_size (int): The number of data points in a batch.
        
    Returns:
        list: A list of mini-batches containing tuples (X_batch, Y_batch).
    """
    m = X.shape[0]
    batches = []
    
    # Shuffle the data synchronously
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Iterate through the shuffled data to create mini-batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        batches.append((X_batch, Y_batch))
        
    return batches
