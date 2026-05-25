#!/usr/bin/env python3
"""
Defines a function that updates the index of a pd.DataFrame
"""
import pandas as pd

def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.
    """
    return df.set_index('Timestamp')
