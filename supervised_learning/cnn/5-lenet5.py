#!/usr/bin/env python3
"""
Builds a modified LeNet-5 architecture using Keras.
"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing the input images
           for the network.

    Returns:
        A K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics.
    """
    init = K.initializers.HeNormal(seed=0)

    # Layer 1: Convolutional layer with 6 kernels of shape 5x5, same padding
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding='same',
                            activation='relu',
                            kernel_initializer=init)(X)

    # Layer 2: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPooling2D(pool_size=2,
                                  strides=2)(conv1)

    # Layer 3: Convolutional layer with 16 kernels of shape 5x5, valid padding
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding='valid',
                            activation='relu',
                            kernel_initializer=init)(pool1)

    # Layer 4: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(pool_size=2,
                                  strides=2)(conv2)

    # Flatten the output for the fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully connected layer with 120 nodes
    dense1 = K.layers.Dense(units=120,
                            activation='relu',
                            kernel_initializer=init)(flatten)

    # Layer 6: Fully connected layer with 84 nodes
    dense2 = K.layers.Dense(units=84,
                            activation='relu',
                            kernel_initializer=init)(dense1)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer=init)(dense2)

    # Build the model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
