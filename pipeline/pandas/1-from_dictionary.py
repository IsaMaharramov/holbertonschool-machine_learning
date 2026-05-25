#!/usr/bin/env python3
"""
Creates a specific pd.DataFrame from a dictionary and saves it to a variable
"""
import pandas as pd

data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
