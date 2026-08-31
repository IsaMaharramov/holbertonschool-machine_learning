#!/usr/bin/env python3
"""
Module 2-gru_cell
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit.
    """

    def __init__(self, i, h, o):
        """
        Constructor.
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        concat_h = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_h, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
