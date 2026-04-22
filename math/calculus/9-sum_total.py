#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    """
    if not isinstance(n, int):
        return None
    if n < 0:
        return None

    # Square pyramidal number formula
    # Result must be an integer
    return (n * (n + 1) * (2 * n + 1)) // 6
