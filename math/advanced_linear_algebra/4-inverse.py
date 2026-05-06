#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """
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


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    n = len(matrix)
    if n == 1:
        return [[1]]

    adj_matrix = []
    for j in range(n):
        adj_row = []
        for i in range(n):
            # Transpose: remove row i and column j to find cofactor C_ij
            # then place it at position (j, i)
            sub = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            sign = (-1) ** (i + j)
            adj_row.append(sign * determinant(sub))
        adj_matrix.append(adj_row)
    return adj_matrix


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    Args:
        matrix: a list of lists whose inverse should be calculated
    Returns:
        The inverse of matrix, or None if matrix is singular
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

    # 1. Calculate the determinant
    det = determinant(matrix)

    # 2. If determinant is 0, the matrix is singular
    if det == 0:
        return None

    # 3. Get the adjugate matrix
    adj = adjugate(matrix)

    # 4. Scale adjugate by 1/det to get the inverse
    inv = [[val / det for val in row] for row in adj]

    return inv
