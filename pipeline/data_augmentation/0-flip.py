#!/usr/bin/env python3
"""Module that contains the flip_image function."""
import tensorflow as tf


def flip_image(image):
    """Flips an image horizontally.

    Args:
        image: a 3D tf.Tensor containing the image to flip.

    Returns:
        The horizontally flipped image tensor.
    """
    return tf.image.flip_left_right(image)
