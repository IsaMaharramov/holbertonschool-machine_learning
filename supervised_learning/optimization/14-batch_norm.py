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
    # Create the base Dense layer without activation
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense_layer(prev)
    
    # Apply Batch Normalization
    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer=tf.keras.initializers.Zeros(),
        gamma_initializer=tf.keras.initializers.Ones()
    )
    Z_norm = batch_norm(Z)
    
    # Apply the activation function
    return activation(Z_norm)
