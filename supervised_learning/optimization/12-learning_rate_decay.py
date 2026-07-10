#!/usr/bin/env python3
"""
Creates a learning rate decay operation in tensorflow using inverse time decay.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.
    
    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate 
                            at which alpha will decay.
        decay_step (int): The number of passes of gradient descent that 
                          should occur before alpha is decayed further.
                          
    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule: The learning 
        rate decay operation.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
