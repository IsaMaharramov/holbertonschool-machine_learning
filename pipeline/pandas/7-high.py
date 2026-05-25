#!/usr/bin/env python3
"""
Defines a function that orders a pd.DataFrame by a metric column
"""


def high(df):
    """
    Sorts the data frame by the High price in descending order.
    """
    return df.sort_values(by='High', ascending=False)
