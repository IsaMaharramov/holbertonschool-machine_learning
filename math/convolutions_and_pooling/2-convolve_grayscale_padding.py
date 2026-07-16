#!/usr/bin/env python3
"""
Module for performing a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: A numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images.
        kernel: A numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution.
        padding: A tuple of (ph, pw) specifying padding for height and width.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Pad images with 0s based on the provided custom padding
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
