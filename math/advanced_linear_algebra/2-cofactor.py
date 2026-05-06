#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    Args:
        matrix: a list of lists whose cofactor matrix should be calculated
    Returns:
        The cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or (n == 1 and len(matrix[0]) == 0):
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Special case for 1x1 matrix
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            # Create submatrix for the minor
            sub = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            # Apply the sign (-1)^(i+j) to the determinant of the submatrix
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * determinant(sub))
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix
