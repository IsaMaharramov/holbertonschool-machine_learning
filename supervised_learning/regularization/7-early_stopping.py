#!/usr/bin/env python3
"""
Module that determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Args:
        cost: The current validation cost of the neural network.
        opt_cost: The lowest recorded validation cost of the network.
        threshold: The threshold used for early stopping.
        patience: The patience count used for early stopping.
        count: The count of how long the threshold has not been met.

    Returns:
        A boolean of whether the network should be stopped early,
        followed by the updated count.
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return (True, count)
    else:
        return (False, count)
