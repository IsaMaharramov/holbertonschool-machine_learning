#!/usr/bin/env python3
"""
Module 3-lstm_cell
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit.
    """

    def __init__(self, i, h, o):
        """
        Constructor.
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_tilde
        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))
        h_next = o * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, c_next, y
