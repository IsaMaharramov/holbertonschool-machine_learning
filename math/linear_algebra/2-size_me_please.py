#!/usr/bin/env python3
"""Module that contains the matrix_shape function"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix and returns it as a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return shape
