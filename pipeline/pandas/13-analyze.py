#!/usr/bin/env python3
"""
Module for computing descriptive statistics
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp
    """
    return df.drop(columns=['Timestamp']).describe()
