#!/usr/bin/env python3
"""
Cleans, transforms, slices, and aggregates data to display a time-series plot
"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# 2. Rename column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# 3. Convert timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Index the data frame on Date
df = df.set_index('Date')

# 5. Missing value imputations
df['Close'] = df['Close'].ffill()
df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 6. Filter data from 2017 and beyond
df = df.loc['2017-01-01':]

# 7. Resample data at daily intervals using specified grouping aggregations
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

print(df)

# Plotting logic
df.plot()
plt.show()
