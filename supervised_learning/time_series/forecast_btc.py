#!/usr/bin/env python3
"""
Forecast BTC using an RNN model.
"""
import tensorflow as tf
import pandas as pd


def windowed_dataset(data, window_size, batch_size, shuffle_buffer):
    """
    Creates a windowed dataset for time series forecasting.

    Args:
        data (np.ndarray): The dataset array.
        window_size (int): The size of the sliding window.
        batch_size (int): The batch size.
        shuffle_buffer (int): The shuffle buffer size.

    Returns:
        tf.data.Dataset: The windowed dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(
        lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Target is the 'Close' price of the next hour (Index 3 of features array)
    dataset = dataset.map(lambda window: (window[:-1], window[-1][3]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    """
    Main function to create, train and validate the model.
    """
    # 1. Load Data
    df = pd.read_csv('preprocessed_data.csv')
    df = df.drop(columns=['Timestamp'])

    data = df.values
    n = len(data)

    # 2. Split Data (70% Train, 20% Val, 10% Test)
    train_data = data[0:int(n * 0.7)]
    val_data = data[int(n * 0.7):int(n * 0.9)]

    # 3. Scale/Normalize the data based on training characteristics
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std

    # 4. Create tf.data pipelines
    window_size = 24
    batch_size = 64
    shuffle_buffer = 1000

    train_set = windowed_dataset(
        train_data, window_size, batch_size, shuffle_buffer)
    val_set = windowed_dataset(
        val_data, window_size, batch_size, shuffle_buffer)

    # 5. Build the LSTM Architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=[window_size, 7]),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model with MSE as requested
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae'])

    # 6. Train the model
    model.fit(train_set, epochs=10, validation_data=val_set)

    # 7. Save the model
    model.save('model.keras')


if __name__ == '__main__':
    main()
