#!/usr/bin/env python3
"""
Module for performing pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: A numpy.ndarray shape (m, h, w, c) containing multiple images.
        kernel_shape: A tuple of (kh, kw) containing the kernel shape.
        stride: A tuple of (sh, sw) containing the strides.
        mode: Indicates the type of pooling ('max' or 'avg').

    Returns:
        A numpy.ndarray containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int(((h - kh) / sh) + 1)
    output_w = int(((w - kw) / sw) + 1)

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return output
