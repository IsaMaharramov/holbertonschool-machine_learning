#!/usr/bin/env python3
"""
Preprocessing data for BTC time series forecasting.
"""
import pandas as pd


def preprocess_data(file_path, output_path="preprocessed_data.csv"):
    """
    Preprocesses the raw coinbase/bitstamp dataset.

    Args:
        file_path (str): The path to the raw CSV data.
        output_path (str): The path to save the preprocessed data.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    df = pd.read_csv(file_path)

    # Convert Unix Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')

    # Drop rows with NaN (typically representing minutes with no trades)
    df = df.dropna()

    # Resample to hourly frequency to match our 24h window/1h forecast goal
    # pandas >= 2.2.0 uses 'h' for hours
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    }
    df_hourly = df.resample('h').agg(agg_dict)

    # Forward fill missing values that may have resulted from resampling
    df_hourly = df_hourly.ffill()

    # Save to csv
    df_hourly.to_csv(output_path)
    return df_hourly


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        preprocess_data(sys.argv[1])
    else:
        print("Usage: ./preprocess_data.py <file_path>")
