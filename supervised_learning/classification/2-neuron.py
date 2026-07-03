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
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Retrieves the weights vector for the neuron."""
        return self.__W

    @property
    def b(self):
        """Retrieves the bias for the neuron."""
        return self.__b

    @property
    def A(self):
        """Retrieves the activated output of the neuron."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): A numpy.ndarray with shape (nx, m) that
                contains the input data.
                - nx is the number of input features to the neuron.
                - m is the number of examples.

        Returns:
            The private attribute __A (the activated output).
        """
        Z = np.matmul(self.__W, X) + self.__b
        
        self.__A = 1 / (1 + np.exp(-Z))
        
        return self.__A
