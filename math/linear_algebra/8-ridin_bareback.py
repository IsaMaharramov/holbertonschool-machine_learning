#!/usr/bin/env python3
"""Module that contains the mat_mul function"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication of two 2D matrices"""
    # Check if matrices can be multiplied (n1 == m2)
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize resulting matrix with zeros (m x p)
    rows_mat1 = len(mat1)
    cols_mat1 = len(mat1[0])
    cols_mat2 = len(mat2[0])

    result = [[0 for _ in range(cols_mat2)] for _ in range(rows_mat1)]

    # Perform multiplication
    for i in range(rows_mat1):
        for j in range(cols_mat2):
            for k in range(cols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
