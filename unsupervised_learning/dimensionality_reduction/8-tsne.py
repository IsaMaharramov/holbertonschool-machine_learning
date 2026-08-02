#!/usr/bin/env python3
"""
t-SNE Complete Transformation Module
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a full t-SNE transformation.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) to be transformed.
        ndims (int): New dimensional representation of X.
        idims (int): Intermediate dimensional representation after PCA.
        perplexity (float): The perplexity for Gaussian distributions.
        iterations (int): The number of iterations for optimization.
        lr (int): The learning rate.

    Returns:
        numpy.ndarray: Y of shape (n, ndim) containing the optimized
            low dimensional transformation of X.
    """
    n, d = X.shape

    # Step 1: Perform PCA for initial dimensionality reduction
    X_pca = pca(X, idims)

    # Step 2: Compute symmetric P affinities
    P = P_affinities(X_pca, perplexity=perplexity)

    # Apply early exaggeration (Multiply by 4 for the first 100 iterations)
    P = P * 4.0

    # Initialize Y with standard normal distribution
    Y = np.random.randn(n, ndims)

    # Initialize the gradient update tracker (momentum steps)
    iY = np.zeros((n, ndims))

    # Step 3: Gradient Descent Optimization Loop
    for i in range(1, iterations + 1):
        # Calculate gradients and Q affinities
        dY, Q = grads(Y, P)

        # Set momentum alpha(t) according to the iteration state
        alpha = 0.5 if i <= 20 else 0.8

        # Update rule: momentum step minus the gradient times learning rate
        iY = alpha * iY - lr * dY
        Y = Y + iY
        
        # Re-center Y by subtracting its mean
        Y = Y - np.mean(Y, axis=0)

        # Evaluate and print the cost every 100 iterations
        if i % 100 == 0:
            # Recompute Q affinities for the newly updated Y to print exact cost
            _, Q_new = grads(Y, P)
            C = cost(P, Q_new)
            print("Cost at iteration {}: {}".format(i, C))

        # Turn off early exaggeration immediately after the 100th iteration ends
        if i == 100:
            P = P / 4.0

    return Y
