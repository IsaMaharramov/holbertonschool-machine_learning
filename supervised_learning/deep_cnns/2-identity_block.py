#!/usr/bin/env python3
"""Identity Block module for ResNet-50"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    for Image Recognition (2015).

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution

    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)

    # Main Path: First component (1x1 Conv)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Main Path: Second component (3x3 Conv)
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Main Path: Third component (1x1 Conv)
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Addition step (Shortcut value + Main path)
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
