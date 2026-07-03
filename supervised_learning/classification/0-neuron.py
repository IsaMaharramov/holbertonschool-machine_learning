#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Initializes the neuron.

        Args:
            nx (int): The number of input features to the neuron.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # The weights vector initialized using a random normal distribution
        self.W = np.random.randn(1, nx)
        # The bias initialized to 0
        self.b = 0
        # The activated output (prediction) initialized to 0
        self.A = 0
