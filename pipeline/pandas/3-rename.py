#!/usr/bin/env python3
"""
Defines a function that renames and modifies a pd.DataFrame
"""
import pandas as pd

def rename(df):
    """
    Renames Timestamp column, converts values to datetime,
    and displays only Datetime and Close columns.
    """
    # Rename column
    df = df.rename(columns={'Timestamp': 'Datetime'})
    # Convert timestamp values to datetime values
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    # Select and display only Datetime and Close
    return df[['Datetime', 'Close']]
