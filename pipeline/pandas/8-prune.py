#!/usr/bin/env python3
"""
Defines a function that scrubs rows missing crucial parameters
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.
    """
    return df.dropna(subset=['Close'])
