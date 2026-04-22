#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    """
    # Strict integer check: if n is not an int, it's not a 'valid number'
    if not isinstance(n, int):
        return None
    if n < 0:
        return None

    # Using the square pyramidal number formula: n(n+1)(2n+1)/6
    # Integer division // ensures we return an int type
    return (n * (n + 1) * (2 * n + 1)) // 6
