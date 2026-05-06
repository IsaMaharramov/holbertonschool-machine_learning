#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    Args:
        matrix: a numpy.ndarray of shape (n, n)
    Returns:
        The string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite
        Returns None if matrix is not a valid matrix or category
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if the matrix is 2D and square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.array_equal(matrix.T, matrix):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # Categorize based on eigenvalue signs
    pos = np.all(eigenvalues > 0)
    neg = np.all(eigenvalues < 0)
    semi_pos = np.all(eigenvalues >= 0)
    semi_neg = np.all(eigenvalues <= 0)

    # Use a small epsilon to detect "zeros" due to floating point precision
    # However, for this project, standard boolean checks usually suffice.
    if pos:
        return "Positive definite"
    if neg:
        return "Negative definite"
    if semi_pos:
        return "Positive semi-definite"
    if semi_neg:
        return "Negative semi-definite"

    # If it has both positive and negative eigenvalues
    any_pos = np.any(eigenvalues > 0)
    any_neg = np.any(eigenvalues < 0)
    if any_pos and any_neg:
        return "Indefinite"

    return None
