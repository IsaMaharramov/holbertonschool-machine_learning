#!/usr/bin/env python3
"""Module that contains the change_brightness function."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image.

    Args:
        image: a 3D tf.Tensor containing the image to change.
        max_delta: maximum amount the image should be brightened or darkened.

    Returns:
        The brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(image, max_delta)
