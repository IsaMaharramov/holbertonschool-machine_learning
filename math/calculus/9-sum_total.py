#!/usr/bin/env python3
"""
Module that calculates the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n using the formula:
    [n(n + 1)(2n + 1)] / 6
    """
    if not isinstance(n, int):
        return None
    if n < 0:
        return None

    # Using integer division // 6 ensures the return type is an int
    # and satisfies the requirement for an integer value.
    return (n * (n + 1) * (2 * n + 1)) // 6
