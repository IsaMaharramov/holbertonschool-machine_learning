#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in TensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.

    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (callable): The activation function to be used.

    Returns:
        tf.Tensor: A tensor of the activated output for the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense_layer(prev)

    # Manually define the trainable parameters gamma and beta
    gamma = tf.Variable(initial_value=tf.ones([n]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([n]), trainable=True)

    # Calculate the mean and variance for the current batch
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Apply batch normalization
    Z_norm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )

    # Return the activated output
    return activation(Z_norm)
