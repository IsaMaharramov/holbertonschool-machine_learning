#!/usr/bin/env python3
"""
Module for performing a convolution on images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images: A numpy.ndarray with shape (m, h, w, c) containing multiple
                images with channels.
        kernel: A numpy.ndarray with shape (kh, kw, c) containing the kernel.
        padding: Tuple of (ph, pw), 'same', or 'valid'.
        stride: Tuple of (sh, sw) containing the strides.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = max(0, int(np.ceil(((h - 1) * sh + kh - h) / 2)))
        pw = max(0, int(np.ceil(((w - 1) * sw + kw - w) / 2)))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    output_h = int(((h + 2 * ph - kh) / sh) + 1)
    output_w = int(((w + 2 * pw - kw) / sw) + 1)

    # Pad images on height and width, but not on m and c
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            image_slice = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return output
