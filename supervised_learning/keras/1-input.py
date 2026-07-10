#!/usr/bin/env python3
"""
Module to build a neural network with the Keras library
using the Functional API.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx (int): Number of input features to the network.
        layers (list): Number of nodes in each layer of the network.
        activations (list): Activation functions used for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        The Keras model.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    L2 = K.regularizers.L2(lambtha)

    for i in range(len(layers)):
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=L2)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
