#!/usr/bin/env python3
"""
Module for calculating the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial
    poly: list of coefficients where index is the power of x
    C: integration constant
    Returns: new list of coefficients or None if inputs are invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for i, coeff in enumerate(poly):
        if not isinstance(coeff, (int, float)):
            return None
        val = coeff / (i + 1)
        if val % 1 == 0:
            val = int(val)
        integral.append(val)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
