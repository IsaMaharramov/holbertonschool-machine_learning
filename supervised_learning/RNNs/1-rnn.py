#!/usr/bin/env python3
"""
Module 1-rnn
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.
    """
    t, m, _ = X.shape
    _, h = h_0.shape
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0
    h_prev = h_0

    for step in range(t):
        h_prev, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_prev
        Y[step] = y

    return H, Y
