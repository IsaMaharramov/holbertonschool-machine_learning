#!/usr/bin/env python3
"""
Module for performing a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: A numpy.ndarray shape (m, h, w, c) containing multiple images.
        kernels: A numpy.ndarray shape (kh, kw, c, nc) containing the kernels.
        padding: Tuple of (ph, pw), 'same', or 'valid'.
        stride: Tuple of (sh, sw) containing the strides.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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

    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                slice_w = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                output[:, i, j, k] = np.sum(slice_w * kernels[:, :, :, k],
                                            axis=(1, 2, 3))

    return output
