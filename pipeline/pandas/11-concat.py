#!/usr/bin/env python3
"""
Defines a function that chains two selected pd.DataFrames together
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes and concatenates rows up to a point from df2 to df1.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_filtered = df2_indexed.loc[:1417411920]

    return pd.concat([df2_filtered, df1_indexed],
                     keys=['bitstamp', 'coinbase'])
