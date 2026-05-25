#!/usr/bin/env python3
"""
Defines a function that concatenates two selected pd.DataFrames
"""
import pandas as pd
index = __import__('10-index').index

def concat(df1, df2):
    """
    Indexes and concatenates rows up to a point from df2 to df1.
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter bitstamp data up to and including timestamp 1417411920
    df2_filtered = df2_indexed.loc[:1417411920]

    # Concatenate df2 rows to the top of df1 with custom keys
    return pd.concat([df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])
