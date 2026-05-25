#!/usr/bin/env python3
"""
Defines a function to generate data summary statistics
"""
import pandas as pd


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.
    """
    return df.drop(columns=['Timestamp']).describe()
