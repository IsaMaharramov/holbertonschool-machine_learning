#!/usr/bin/env python3
"""
Forward propagation over a pooling layer of a neural network
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
        kernel_shape: a tuple of (kh, kw) containing the size of the kernel
        stride: a tuple of (sh, sw) containing the strides for the pooling
        mode: a string containing either 'max' or 'avg'

    Returns:
        The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int(1 + (h_prev - kh) / sh)
    w_new = int(1 + (w_prev - kw) / sw)

    A = np.zeros((m, h_new, w_new, c_prev))

    for h in range(h_new):
        for w in range(w_new):
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            slice_A = A_prev[:, v_start:v_end, h_start:h_end, :]

            if mode == 'max':
                A[:, h, w, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, h, w, :] = np.mean(slice_A, axis=(1, 2))

    return A
