#!/usr/bin/env python3
"""
Defines a function that sorts a pd.DataFrame by a column value
"""
import pandas as pd


def high(df):
    """
    Sorts the data frame by the High price in descending order.
    """
    return df.sort_values(by='High', ascending=False)
