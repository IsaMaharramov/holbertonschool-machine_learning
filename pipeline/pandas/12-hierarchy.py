#!/usr/bin/env python3
"""
Defines a function that builds a specific hierarchical MultiIndex dataframe
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates segments and swaps index levels chronologically.
    """
    # Index both tables on Timestamp
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter both dataframes within timestamps 1417411980 to 1417417980
    start, end = 1417411980, 1417417980
    df1_filtered = df1_indexed.loc[start:end]
    df2_filtered = df2_indexed.loc[start:end]

    # Concatenate keeping source tags as the top level first
    concat_df = pd.concat([df2_filtered, df1_filtered],
                          keys=['bitstamp', 'coinbase'])

    # Swap index levels to make Timestamp the first level (level 0)
    concat_df = concat_df.swaplevel(0, 1)

    # Ensure the dataframe layout is strictly sorted chronologically by index
    return concat_df.sort_index()
