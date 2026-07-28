#!/usr/bin/env python3
"""Module that contains the crop_image function."""
import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image.

    Args:
        image: a 3D tf.Tensor containing the image to crop.
        size: a tuple containing the size of the crop.

    Returns:
        The randomly cropped image tensor.
    """
    if len(size) == 2:
        size = (size[0], size[1], image.shape[-1])

    return tf.image.random_crop(image, size=size)
