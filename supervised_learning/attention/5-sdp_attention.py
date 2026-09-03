#!/usr/bin/env python3
"""
Scaled Dot Product Attention module
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    :param Q: tensor with shape (..., seq_len_q, dk) containing queries
    :param K: tensor with shape (..., seq_len_v, dk) containing keys
    :param V: tensor with shape (..., seq_len_v, dv) containing values
    :param mask: optional tensor broadcastable into shape (..., seq_len_q, seq_len_v)
    :return: output, weights
             - output: tensor with shape (..., seq_len_q, dv)
             - weights: tensor with shape (..., seq_len_q, seq_len_v)
    """
    # Matmul Q and K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by square root of key dimension dk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add mask to scaled tensor (mask elements multiplied by -1e9)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax over last dimension (seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Matmul attention weights and V
    output = tf.matmul(weights, V)

    return output, weights
