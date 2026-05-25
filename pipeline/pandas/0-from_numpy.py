#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with capitalized
    alphabetical labels for columns.
    """
    num_cols = array.shape[1]
    col_labels = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=col_labels)
