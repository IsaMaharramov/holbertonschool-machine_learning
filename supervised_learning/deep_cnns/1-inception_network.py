#!/usr/bin/env python3
"""Inception Network module"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in
    Going Deeper with Convolutions (2014).

    Returns: the keras model
    """
    init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1: Initial Conv and Max Pooling
    X = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2),
        padding='same', activation='relu', kernel_initializer=init
    )(X_input)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(X)

    # Stage 2: 1x1 Conv, 3x3 Conv, and Max Pooling
    X = K.layers.Conv2D(
        filters=64, kernel_size=(1, 1), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer=init
    )(X)
    X = K.layers.Conv2D(
        filters=192, kernel_size=(3, 3), strides=(1, 1),
        padding='same', activation='relu', kernel_initializer=init
    )(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(X)

    # Stage 3: Inception Blocks (3a, 3b) and Max Pooling
    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    X = inception_block(X, [128, 128, 192, 32, 96, 64])
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(X)

    # Stage 4: Inception Blocks (4a, 4b, 4c, 4d, 4e) and Max Pooling
    X = inception_block(X, [192, 96, 208, 16, 48, 64])
    X = inception_block(X, [160, 112, 224, 24, 64, 64])
    X = inception_block(X, [128, 128, 256, 24, 64, 64])
    X = inception_block(X, [112, 144, 288, 32, 64, 64])
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(X)

    # Stage 5: Inception Blocks (5a, 5b)
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = inception_block(X, [384, 192, 384, 48, 128, 128])

    # Average Pooling, Dropout, and Dense Classifier
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1), padding='valid'
    )(X)
    X = K.layers.Dropout(rate=0.4)(X)
    X = K.layers.Dense(
        units=1000, activation='softmax', kernel_initializer=init
    )(X)

    model = K.models.Model(inputs=X_input, outputs=X)
    return model
