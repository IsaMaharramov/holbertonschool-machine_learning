#!/usr/bin/env python3
"""
Module for concatenating N-dimensional matrices along a specific axis.
"""


def get_shape(matrix):
    """Returns the shape of a matrix as a list of integers."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1: The first matrix (list of lists).
        mat2: The second matrix (list of lists).
        axis: The axis along which to concatenate.

    Returns:
        A new concatenated matrix, or None if concatenation is impossible.
    """
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check if they have the same number of dimensions
    if len(shape1) != len(shape2):
        return None

    # For the target axis, any length is fine.
    # For all other axes, the lengths must be identical.
    for i in range(len(shape1)):
        if i != axis:
            if shape1[i] != shape2[i]:
                return None

    def recurse_cat(m1, m2, current_axis):
        """Helper to navigate to the correct axis and concatenate."""
        if current_axis == axis:
            return m1 + m2

        # If we aren't at the axis yet, we must iterate through the elements
        new_matrix = []
        for i in range(len(m1)):
            res = recurse_cat(m1[i], m2[i], current_axis + 1)
            new_matrix.append(res)
        return new_matrix

    return recurse_cat(mat1, mat2, 0)
