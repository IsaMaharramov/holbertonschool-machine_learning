#!/usr/bin/env python3
"""
Defines a function that drops missing data points from a pd.DataFrame
"""
import pandas as pd


def prune(df):
    """
    Removes any entries where Close has NaN values.
    """
    return df.dropna(subset=['Close'])
