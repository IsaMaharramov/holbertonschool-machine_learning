#!/usr/bin/env python3
"""
Defines a function that performs matrix multiplication
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication on two numpy.ndarrays
    Args:
        mat1: first numpy.ndarray
        mat2: second numpy.ndarray
    Returns:
        A new numpy.ndarray containing the matrix product
    """
    return np.matmul(mat1, mat2)
