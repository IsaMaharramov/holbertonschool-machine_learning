#!/usr/bin/env python3
"""
Module for calculating the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    poly: list of coefficients where index is the power of x
    Returns: new list of coefficients or None if poly is invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check if all elements in the list are numbers
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # If the polynomial is just a constant (e.g., [5]), derivative is [0]
    if len(poly) == 1:
        return [0]

    derivative = []
    # Skip index 0 (constant term) and calculate the rest
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    # If the entire result is 0 (e.g., derivative of [5, 0, 0])
    if not any(derivative):
        return [0]

    # Remove trailing zeros to keep the list as small as possible
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
