#!/usr/bin/env python3
"""
Defines a function that calculates descriptive statistics for a dataframe
"""
import pandas as pd


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.
    """
    # Drop Timestamp before executing describe
    return df.drop(columns=['Timestamp']).describe()
