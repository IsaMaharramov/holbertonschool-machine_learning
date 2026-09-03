#!/usr/bin/env python3
"""
RNN Decoder Module
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


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
        self.attention = SelfAttention(units)

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
        context, weights = self.attention(s_prev, hidden_states)
        x = self.embedding(x)

        # Expand context vector to shape (batch, 1, units)
        context_expanded = tf.expand_dims(context, 1)

        # Concatenate context vector with embedded x
        x = tf.concat([context_expanded, x], axis=-1)

        # Pass concatenated vector to the GRU
        output, s = self.gru(x)

        # Reshape output to (batch, units) for the dense layer
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass through Dense layer to produce vocabulary logits
        y = self.F(output)

        return y, s
