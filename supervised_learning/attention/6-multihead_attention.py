#!/usr/bin/env python3
"""
Multi Head Attention module
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer
    to perform multi head attention
    """

    def __init__(self, dm, h):
        """
        Class constructor

        :param dm: integer representing the dimensionality of the model
        :param h: integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth) and transposes the result
        to shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """
        Executes the multi head attention

        :param Q: tensor of shape (batch, seq_len_q, dk) (Query)
        :param K: tensor of shape (batch, seq_len_v, dk) (Key)
        :param V: tensor of shape (batch, seq_len_v, dv) (Value)
        :param mask: optional mask tensor or None
        :return: output, weights
                 - output: tensor of shape (batch, seq_len_q, dm)
                 - weights: tensor of shape (batch, h, seq_len_q, seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        # Pass through linear projections
        q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        k = self.Wk(K)  # (batch_size, seq_len_v, dm)
        v = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)  # (batch_size, h, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, h, seq_len_v, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, h, seq_len_v, depth)

        # Calculate scaled dot product attention
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        # Concatenate heads back together
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # Pass through final linear layer
        output = self.linear(concat_attention)

        return output, attention_weights
