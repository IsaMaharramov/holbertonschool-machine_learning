#!/usr/bin/env python3
"""
Module to build a neural network with the Keras library
using the Sequential API.
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
    model = K.Sequential()
    L2 = K.regularizers.L2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2,
                                     input_shape=(nx,)))
        else:
            # Dropout rate is the fraction to drop (1 - keep_prob)
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2))

    return model
