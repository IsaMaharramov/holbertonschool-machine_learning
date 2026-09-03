#!/usr/bin/env python3
"""
RNN Decoder Module
"""
import tensorflow as tf


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class that decodes for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor for RNNDecoder
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call method for RNNDecoder
        Args:
            x: tensor of shape (batch, 1) with previous word as index
            s_prev: tensor of shape (batch, units) previous hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
        Returns:
            y: tensor of shape (batch, vocab) with the output word one-hot
            s: tensor of shape (batch, units) with new decoder hidden state
        """
        SelfAttention = __import__('1-self_attention').SelfAttention
        attention = SelfAttention(s_prev.shape[1])

        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)

        # Expand context vector to shape (batch, 1, units)
        context_expanded = tf.expand_dims(context, 1)

        # Concatenate context vector with x
        x = tf.concat([context_expanded, x], axis=-1)

        # Pass concatenated vector to the GRU
        output, s = self.gru(x)

        # Reshape the output to (batch, units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass the output through the Dense layer to get vocabulary scores
        y = self.F(output)

        return y, s
