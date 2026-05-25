#!/usr/bin/env python3
"""
Defines a function that cuts a pd.DataFrame into pieces
"""
import pandas as pd


def slice(df):
    """
    Extracts columns High, Low, Close, and Volume_(BTC)
    and selects every 60th row.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
