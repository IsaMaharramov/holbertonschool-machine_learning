#!/usr/bin/env python3
"""Projection Block module for ResNet-50"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
            """
            Builds a projection block as described in Deep Residual Learning
            for Image Recognition (2015).

            A_prev is the output from the previous layer
            filters is a tuple or list containing F11, F3, F12, respectively:
                F11 is the number of filters in the first 1x1 convolution
                F3 is the number of filters in the 3x3 convolution
                F12 is the number of filters in the second 1x1 convolution
            s is the stride of the first convolution in both the main path
                and the shortcut connection

            Returns: the activated output of the projection block
            """
            F11, F3, F12 = filters
            init = K.initializers.he_normal(seed=0)

            # Main Path: First component (1x1 Conv)
            X = K.layers.Conv2D(
                filters=F11,
                kernel_size=(1, 1),
                strides=(s, s),
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

            # Shortcut Path (1x1 Conv)
            X_shortcut = K.layers.Conv2D(
                filters=F12,
                kernel_size=(1, 1),
                strides=(s, s),
                padding='valid',
                kernel_initializer=init
            )(A_prev)
            X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

            # Addition step (Shortcut value + Main path)
            X = K.layers.Add()([X, X_shortcut])
            X = K.layers.Activation('relu')(X)

            return X
