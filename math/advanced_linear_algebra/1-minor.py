#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a matrix
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


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    Args:
        matrix: a list of lists whose minor matrix should be calculated
    Returns:
        The minor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Handle the special case for 1x1 matrix per example output
    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            # Create submatrix by removing row i and column j
            sub = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            row_minors.append(determinant(sub))
        minor_matrix.append(row_minors)

    return minor_matrix
