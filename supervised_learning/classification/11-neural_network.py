#!/usr/bin/env python3
"""Module that defines a neural network with one hidden layer."""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing
    binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the Neural Network.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes found in the hidden layer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Retrieves the weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Retrieves the bias for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Retrieves the activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Retrieves the weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Retrieves the bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Retrieves the activated output for the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            tuple: The private attributes __A1 and __A2, respectively.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels for the input data.
            A (numpy.ndarray): Activated output of the neuron.

        Returns:
            The logistic regression cost.
        """
        m = Y.shape[1]
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss)
        return cost
