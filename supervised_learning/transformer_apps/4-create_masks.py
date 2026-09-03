#!/usr/bin/env python3
"""
Creates masks for training/validation in Transformer Applications
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) containing
                the input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out) containing
                the target sentence

    Returns:
        encoder_mask, combined_mask, decoder_mask
    """
    # Encoder padding mask (used in the encoder)
    # Shape: (batch_size, 1, 1, seq_len_in)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask (used in the 2nd attention block in the decoder)
    # Shape: (batch_size, 1, 1, seq_len_in)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Lookahead mask (used in the 1st attention block in the decoder)
    # Shape: (seq_len_out, seq_len_out)
    seq_len_out = tf.shape(target)[1]
    lookahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    # Target padding mask
    # Shape: (batch_size, 1, 1, seq_len_out)
    dec_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_mask = dec_target_mask[:, tf.newaxis, tf.newaxis, :]

    # Combined mask (max between lookahead and target padding)
    # Shape: (batch_size, 1, seq_len_out, seq_len_out)
    combined_mask = tf.maximum(dec_target_mask, lookahead_mask)

    return encoder_mask, combined_mask, decoder_mask
