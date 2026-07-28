#!/usr/bin/env python3
"""
Back propagation over a pooling layer of a neural network
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the output of the pooling layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
            output of the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel
        stride: tuple of (sh, sw) containing the strides for the pooling
        mode: string containing either 'max' or 'avg', indicating whether to
            perform maximum or average pooling

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize dA_prev with zeros in the same shape as A_prev
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    # Define the corners of the current slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        # Get the current slice from A_prev
                        a_prev_slice = A_prev[i, vert_start:vert_end,
                                              horiz_start:horiz_end, ch]

                        # Create a boolean mask of the max value in the slice
                        mask = (a_prev_slice == np.max(a_prev_slice))

                        # Distribute the gradient only to the max value(s)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += (mask *
                                                               dA[i, h, w, ch])

                    elif mode == 'avg':
                        # Get the gradient for this position
                        da = dA[i, h, w, ch]

                        # Distribute it equally across the kernel window
                        shape = (kh, kw)
                        average = da / (kh * kw)
                        a = np.ones(shape) * average

                        # Add the distributed gradient to dA_prev
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, ch] += a

    return dA_prev
