#!/usr/bin/env python3
"""
Module that calculates the cost of a neural network with L2 regularization
using TensorFlow
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    
    Args:
        cost: A tensor containing the cost of the network without L2 reg.
        model: A Keras model that includes layers with L2 regularization.
        
    Returns:
        A tensor containing the total cost for each layer of the network,
        accounting for L2 regularization.
    """
    # model.losses returns a list of regularization losses for each layer.
    # Stacking it converts it into a single tensor for broadcasting.
    return cost + tf.stack(model.losses)
