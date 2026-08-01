#!/usr/bin/env python3
"""Dense Block module"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional
    Networks using bottleneck layers for DenseNet-B.

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block

    Returns: The concatenated output of each layer within the Dense Block
             and the number of filters within the concatenated outputs
    """
    init = K.initializers.he_normal(seed=0)

    for i in range(layers):
        # Bottleneck: 1x1 Convolution mapping to 4 * growth_rate filters
        H = K.layers.BatchNormalization(axis=3)(X)
        H = K.layers.Activation('relu')(H)
        H = K.layers.Conv2D(
            filters=(4 * growth_rate),
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=init
        )(H)

        # 3x3 Convolution mapping to growth_rate filters
        H = K.layers.BatchNormalization(axis=3)(H)
        H = K.layers.Activation('relu')(H)
        H = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=init
        )(H)

        # Concatenate block output with previous layer output
        X = K.layers.Concatenate(axis=3)([X, H])
        nb_filters += growth_rate

    return X, nb_filters
