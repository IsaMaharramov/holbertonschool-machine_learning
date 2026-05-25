#!/usr/bin/env python3
"""
Defines a function that handles missing values inside a pd.DataFrame
"""
import pandas as pd


def fill(df):
    """
    Fills specific missing values across financial metrics columns.
    """
    # Remove Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # Fill missing values in Close with previous row value
    df['Close'] = df['Close'].ffill()

    # Fill Open, High, Low missing values with the corresponding row's Close
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])

    # Set missing volume tracks to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
