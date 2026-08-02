#!/usr/bin/env python3
"""
P Affinities Module for t-SNE
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) to be transformed.
        tol (float): Maximum tolerance allowed for Shannon entropy difference.
        perplexity (float): Perplexity that all distributions should have.

    Returns:
        numpy.ndarray: P of shape (n, n) containing symmetric P affinities.
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        # Extract the pairwise distances excluding the point itself
        Di = np.concatenate((D[i, :i], D[i, i+1:]))

        # Isolate the beta for the current point
        beta = betas[i].copy()
        beta_min = None
        beta_max = None

        # Perform binary search to find the correct beta value
        while True:
            Hi, Pi = HP(Di, beta)
            H_diff = Hi - H

            # Check if entropy falls within the tolerance
            if np.abs(H_diff) <= tol:
                break

            if H_diff > 0:
                # Entropy is too high (distribution is too spread out)
                # -> increase beta
                beta_min = beta[0]
                if beta_max is None:
                    beta[0] *= 2.0
                else:
                    beta[0] = (beta[0] + beta_max) / 2.0
            else:
                # Entropy is too low (distribution is too concentrated)
                # -> decrease beta
                beta_max = beta[0]
                if beta_min is None:
                    beta[0] /= 2.0
                else:
                    beta[0] = (beta[0] + beta_min) / 2.0

        # Save the found beta and append the computed P affinities
        betas[i] = beta
        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]

    # Make the P affinities symmetric and scale by (2 * n)
    P = (P + P.T) / (2 * n)

    return P
