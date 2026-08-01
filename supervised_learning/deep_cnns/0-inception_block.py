#!/usr/bin/env python3
"""Inception Block module"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with Convolutions.

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3, F5R, F5, FPP:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 conv before the 3x3 conv
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 conv before the 5x5 conv
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 conv after the max pooling

    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=0)

    # 1x1 Convolution Path
    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    # 3x3 Convolution Path
    conv3r = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv3r)

    # 5x5 Convolution Path
    conv5r = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv5r)

    # Max Pooling Path
    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)
    convpp = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(pool)

    # Concatenate all paths along the channel axis (axis=-1 or 3)
    output = K.layers.Concatenate(axis=3)([conv1, conv3, conv5, convpp])

    return output
