#!/usr/bin/env python3
"""
Defines a function that handles empty values inside a pd.DataFrame
"""


def fill(df):
    """
    Fills specific missing values across financial metrics columns.
    """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].ffill()

    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
