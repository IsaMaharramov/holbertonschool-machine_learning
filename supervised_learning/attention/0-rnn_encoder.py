#!/usr/bin/env python3
"""
RNN Encoder Module
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class that encodes for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor for RNNEncoder
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns:
            A tensor of shape (batch, units) containing initialized states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Call method for the RNN Encoder
        Args:
            x: tensor of shape (batch, input_seq_len) containing word indices
            initial: tensor of shape (batch, units) with initial hidden state
        Returns:
            outputs: tensor of shape (batch, input_seq_len, units)
            hidden: tensor of shape (batch, units) with the last hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
