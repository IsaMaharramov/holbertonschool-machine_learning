#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    Args:
        matrix: a list of lists whose determinant should be calculated
    Returns:
        The determinant of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Special case for 0x0 matrix
    if matrix == [[]]:
        return 1

    n = len(matrix)

    # Check if the matrix is square
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix (optimization)
    if n == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    # Recursive Laplace expansion
    det = 0
    for j in range(n):
        # Create submatrix by removing the 0-th row and j-th column
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # Alternate signs: (-1)^i+j where i=0
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(sub_matrix)

    return det
