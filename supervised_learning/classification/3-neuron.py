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
        # Linear combination: Z = WX + b
        Z = np.matmul(self.__W, X) + self.__b
        
        # Sigmoid activation function: A = 1 / (1 + e^-Z)
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): A numpy.ndarray with shape (1, m) that contains
                the correct labels for the input data.
            A (numpy.ndarray): A numpy.ndarray with shape (1, m) containing
                the activated output of the neuron for each example.

        Returns:
            The logistic regression cost.
        """
        m = Y.shape[1]
        
        # Logistic Regression Cost Function (Cross-Entropy Loss)
        # Using 1.0000001 - A to prevent np.log(0) which causes division by zero/NaN errors
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        
        return cost
