#!/usr/bin/env python3
"""
Module containing the np_slice function for dynamic matrix slicing.
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.

    Args:
        matrix: A numpy.ndarray to slice.
        axes: A dictionary where the key is the axis and the value
              is a tuple representing the slice.

    Returns:
        A new numpy.ndarray representing the slice.
    """
    # Create a list of 'slice(None)' objects for every dimension
    # slice(None) is equivalent to ':' in standard indexing
    slices = [slice(None)] * len(matrix.shape)

    # Replace the default 'slice(None)' with the specific tuple from axes
    for axis, slice_tuple in axes.items():
        # slice(*slice_tuple) unpacks (start, stop, step) into the slice object
        slices[axis] = slice(*slice_tuple)

    # Use the tuple of slice objects to index the matrix
    return matrix[tuple(slices)]
