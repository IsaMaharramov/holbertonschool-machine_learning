#!/usr/bin/env python3
"""Module that contains the change_hue function."""
import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image.

    Args:
        image: a 3D tf.Tensor containing the image to change.
        delta: the amount the hue should change.

    Returns:
        The hue-adjusted image tensor.
    """
    return tf.image.adjust_hue(image, delta)
