#!/usr/bin/env python3
"""
Forward propagation over a convolutional layer of a neural network
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a
    neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        activation: an activation function applied to the convolution
        padding: a string that is either 'same' or 'valid'
        stride: a tuple of (sh, sw) containing the strides for the convolution

    Returns:
        The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
        ph = max(ph, 0)
        pw = max(pw, 0)
    else:
        ph, pw = 0, 0

    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')

    h_new = int((h_prev - kh + 2 * ph) / sh) + 1
    w_new = int((w_prev - kw + 2 * pw) / sw) + 1

    Z = np.zeros((m, h_new, w_new, c_new))

    for h in range(h_new):
        for w in range(w_new):
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            slice_A = A_pad[:, v_start:v_end, h_start:h_end, :]

            for c in range(c_new):
                Z[:, h, w, c] = np.sum(slice_A * W[:, :, :, c], axis=(1, 2, 3))

    return activation(Z + b)
