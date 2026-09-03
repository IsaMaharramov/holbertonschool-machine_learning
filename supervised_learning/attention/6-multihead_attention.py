#!/usr/bin/env python3
"""
Multi Head Attention module
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class that performs multi head attention
    """

    def __init__(self, dm, h):
        """
        Class constructor
        Args:
            dm: an integer representing the dimensionality of the model
            h: an integer representing the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Calls the layer
        Args:
            Q is a tensor of shape (batch, seq_len_q, dk)
            K is a tensor of shape (batch, seq_len_v, dk)
            V is a tensor of shape (batch, seq_len_v, dv)
            mask is always None
        Returns:
            output, weights
        """
        batch_size = tf.shape(Q)[0]

        # Pass inputs through linear layers
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Split heads: (batch_size, seq_len, h, depth)
        q = tf.reshape(q, (batch_size, -1, self.h, self.depth))
        k = tf.reshape(k, (batch_size, -1, self.h, self.depth))
        v = tf.reshape(v, (batch_size, -1, self.h, self.depth))

        # Transpose to get shape: (batch_size, h, seq_len, depth)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Calculate Scaled Dot Product Attention
        # output shape: (batch_size, h, seq_len_q, depth)
        # weights shape: (batch_size, h, seq_len_q, seq_len_v)
        output, weights = sdp_attention(q, k, v, mask)

        # Transpose back to (batch_size, seq_len_q, h, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])

        # Concatenate heads: (batch_size, seq_len_q, dm)
        concat_attention = tf.reshape(output, (batch_size, -1, self.dm))

        # Pass through the final linear layer
        output = self.linear(concat_attention)

        return output, weights
