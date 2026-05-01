#!/usr/bin/env python3
"""
Defines a function that concatenates two matrices along a specific axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Args:
        mat1: first numpy.ndarray
        mat2: second numpy.ndarray
        axis: the axis along which the matrices should be concatenated
    Returns:
        A new numpy.ndarray containing the concatenated matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
