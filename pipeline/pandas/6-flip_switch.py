#!/usr/bin/env python3
"""
Defines a function that reorders and transposes a pd.DataFrame
"""
import pandas as pd


def flip_switch(df):
    """
    Sorts data in reverse chronological order and transposes it.
    """
    # Reverse order using negative slicing step on index level, then transpose
    return df.iloc[::-1].T
