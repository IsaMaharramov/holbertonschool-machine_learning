#!/usr/bin/env python3
"""
Module 4-deep_rnn
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for step in range(t):
        x = X[step]
        for layer_idx, cell in enumerate(rnn_cells):
            h_prev = H[step, layer_idx]
            h_next, y = cell.forward(h_prev, x)
            H[step + 1, layer_idx] = h_next
            x = h_next
        Y[step] = y

    return H, Y
