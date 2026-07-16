#!/usr/bin/env python3
"""
Module that creates a layer of a neural network using dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: A tensor containing the output of the previous layer.
        n: The number of nodes the new layer should contain.
        activation: The activation function for the new layer.
        keep_prob: The probability that a node will be kept.
        training: Boolean indicating whether the model is in training mode.

    Returns:
        The output of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))

    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=init)

    dropout = tf.keras.layers.Dropout(rate=(1 - keep_prob))

    return dropout(layer(prev), training=training)
