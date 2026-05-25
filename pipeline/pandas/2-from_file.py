#!/usr/bin/env python3
"""
Defines a function that loads data from a file as a pd.DataFrame
"""
import pandas as pd

def from_file(filename, delimiter):
    """
    Loads data from a file using a given delimiter
    """
    return pd.read_csv(filename, sep=delimiter)
