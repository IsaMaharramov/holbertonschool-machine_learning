# Time Series Forecasting

## Project Description
This repository contains a full pipeline designed to build, train, and validate a Recurrent Neural Network (RNN) to forecast the price of Bitcoin (BTC). The model uses historical 24-hour windows to predict the subsequent hour's closing price. 

## Files
* `preprocess_data.py`: A data engineering script that intakes a raw minute-level dataset (`.csv`), converts Unix timestamps to standard DateTimes, isolates relevant missing intervals, and aggregates the values on an hourly frequency.
* `forecast_btc.py`: Handles model building and evaluation. Standardizes data splits to prevent data leakage, instantiates the `tf.data.Dataset` mapping the sequential sliding windows (24 feature inputs to 1 prediction target window limit), and trains the sequence via an LSTM-based network relying on `MSE` parameters.

## Usage
First, run the data cleaner by providing it with a valid bitstamp/coinbase `.csv` dataset:
```bash
./preprocess_data.py bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv
```
Next, run the main model trainer module which interacts with the generated preprocessed_data.csv:
```bash
./forecast_btc.py
```