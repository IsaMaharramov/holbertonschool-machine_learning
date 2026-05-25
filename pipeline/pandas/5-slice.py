#!/usr/bin/env python3
"""
Defines a function that slices a pd.DataFrame
"""
import pandas as pd

def slice(df):
    """
    Extracts columns High, Low, Close, and Volume_(BTC)
    and selects every 60th row.
    """
    # Using data frame syntax to pick columns and step slice rows
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
