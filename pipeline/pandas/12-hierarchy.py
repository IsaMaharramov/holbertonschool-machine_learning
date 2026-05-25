#!/usr/bin/env python3
"""
Defines a function to construct a hierarchical MultiIndex dataset
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates data sections and swaps index levels chronologically.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    start, end = 1417411980, 1417417980
    df1_filtered = df1_indexed.loc[start:end]
    df2_filtered = df2_indexed.loc[start:end]

    concat_df = pd.concat([df2_filtered, df1_filtered],
                          keys=['bitstamp', 'coinbase'])

    concat_df = concat_df.swaplevel(0, 1)
    return concat_df.sort_index()
