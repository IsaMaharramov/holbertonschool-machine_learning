#!/usr/bin/env python3
"""
Defines a function that converts parts of a pd.DataFrame into a numpy array
"""
import pandas as pd

def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them into a numpy ndarray.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
