#!/usr/bin/env python3
"""
Module for recursive N-dimensional matrix addition.
"""


def add_matrices(mat1, mat2):
    """
    Adds two N-dimensional matrices.

    Args:
        mat1: The first matrix (list of lists... of ints/floats).
        mat2: The second matrix (list of lists... of ints/floats).

    Returns:
        A new matrix containing the sum, or None if shapes don't match.
    """
    # Check if both are lists (matrix/sub-matrix)
    if isinstance(mat1, list) and isinstance(mat2, list):
        # Shape check: dimensions must match at every level
        if len(mat1) != len(mat2):
            return None

        new_matrix = []
        for i in range(len(mat1)):
            res = add_matrices(mat1[i], mat2[i])
            # If any recursive call returns None, the whole thing is None
            if res is None:
                return None
            new_matrix.append(res)
        return new_matrix

    # If they are not lists, they are scalars (base case)
    # Check if only one is a list (shape mismatch)
    if isinstance(mat1, list) or isinstance(mat2, list):
        return None

    return mat1 + mat2
