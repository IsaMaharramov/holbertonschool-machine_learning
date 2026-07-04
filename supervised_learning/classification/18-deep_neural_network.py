#!/usr/bin/env python3
"""
Module that defines a DeepNeuralNetwork -> binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Args:
            nx (int): The number of input features.
            layers (list): A list representing the number of nodes in each
                layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine the size of the previous layer
            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization -> weights
            self.__weights['W' + str(i + 1)] = np.random.randn(
                layers[i], prev_size) * np.sqrt(2 / prev_size)

            # Zero initialization -> biases
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter -> the number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Getter -> the cache dictionary holding intermediary values."""
        return self.__cache

    @property
    def weights(self):
        """Getter -> the weights dictionary holding weights and biases."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Array with shape (nx, m) containing input data.

        Returns:
            The output of the neural network and the cache, respectively.
        """
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]

            # Linear forward
            Z = np.matmul(W, A_prev) + b

            # Sigmoid activation
            A = 1 / (1 + np.exp(-Z))

            # Store the activated output
            self.__cache['A' + str(i)] = A

        return self.__cache['A' + str(self.__L)], self.__cache
