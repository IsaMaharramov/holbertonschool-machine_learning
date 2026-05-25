#!/usr/bin/env python3
"""
Defines a function that transposes a pd.DataFrame object
"""
import pandas as pd


def flip_switch(df):
    """
    Sorts data in reverse chronological order and transposes it.
    """
    return df.iloc[::-1].T
