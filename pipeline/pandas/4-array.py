#!/usr/bin/env python3
"""
Module to convert dataframe to numpy array
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them into a numpy ndarray.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
