#!/usr/bin/env python3
"""
Defines a function that transposes a numpy.ndarray
"""


def np_transpose(matrix):
    """
    Transposes a numpy.ndarray
    Args:
        matrix: the numpy.ndarray to transpose
    Returns:
        A new numpy.ndarray that is the transpose of matrix
    """
    return matrix.transpose()
