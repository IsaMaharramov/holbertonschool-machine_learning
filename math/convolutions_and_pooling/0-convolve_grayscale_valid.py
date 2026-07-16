#!/usr/bin/env python3
"""
Module for performing a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

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

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
