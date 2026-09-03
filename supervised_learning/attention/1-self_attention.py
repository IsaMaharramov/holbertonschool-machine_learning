#!/usr/bin/env python3
"""
Self Attention Module
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention class to calculate the attention for machine translation
    """
    def __init__(self, units):
        """
        Constructor for SelfAttention
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Call method for SelfAttention
        Args:
            s_prev: tensor of shape (batch, units) containing the previous
                    decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                           containing the outputs of the encoder
        Returns:
            context: tensor of shape (batch, units) with the context vector
            weights: tensor of shape (batch, input_seq_len, 1) with weights
        """
        # Expand s_prev to shape (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Calculate attention scores
        # shape: (batch, input_seq_len, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) +
                                  self.U(hidden_states)))

        # Calculate attention weights
        weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
