#!/usr/bin/env python3
"""
Module for performing a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: A numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images.
        kernel: A numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding to ensure output size matches input size
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    # Pad images with 0s on the height and width dimensions
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
