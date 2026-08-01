#!/usr/bin/env python3
"""Transition Layer module"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
    Networks, implementing compression as used in DenseNet-C.

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    Returns: The output of the transition layer and the number of filters
             within the output, respectively
    """
    init = K.initializers.he_normal(seed=0)
    compressed_filters = int(nb_filters * compression)

    # Pre-activation + 1x1 Convolution for compression
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=compressed_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)

    # 2x2 Average Pooling layer to reduce spatial dimensions
    X = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(X)

    return X, compressed_filters
