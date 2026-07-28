#!/usr/bin/env python3
"""
Back propagation over a convolutional layer of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the unactivated output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing kernels
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing biases
        padding: a string that is either 'same' or 'valid'
        stride: a tuple of (sh, sw) containing the strides for the convolution

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
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
    dA_pad = np.zeros(A_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for h in range(h_new):
        for w in range(w_new):
            for c in range(c_new):
                v_start = h * sh
                v_end = v_start + kh
                h_start = w * sw
                h_end = h_start + kw

                slice_A = A_pad[:, v_start:v_end, h_start:h_end, :]
                dz = dZ[:, h, w, c].reshape(m, 1, 1, 1)

                dA_pad[:, v_start:v_end, h_start:h_end, :] += (
                    W[np.newaxis, :, :, :, c] * dz
                )
                dW[:, :, :, c] += np.sum(slice_A * dz, axis=0)

    dA_prev = dA_pad[:, ph:ph+h_prev, pw:pw+w_prev, :]

    return dA_prev, dW, db
