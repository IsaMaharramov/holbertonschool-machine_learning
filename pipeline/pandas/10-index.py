#!/usr/bin/env python3
"""
Defines a function that configures the index of a pd.DataFrame
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.
    """
    return df.set_index('Timestamp')
