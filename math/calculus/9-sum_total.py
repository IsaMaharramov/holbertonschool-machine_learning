#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    """
    # Check if n is specifically an integer type
    if type(n) is not int:
        return None
    if n < 0:
        return None

    # Square pyramidal number formula: n(n + 1)(2n + 1) / 6
    # Integer division // ensures the return is an integer
    return (n * (n + 1) * (2 * n + 1)) // 6
