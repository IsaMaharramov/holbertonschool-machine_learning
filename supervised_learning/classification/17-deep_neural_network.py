#!/usr/bin/env python3
"""
Module that defines the DeepNeuralNetwork class.
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

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list or is an empty list.
            TypeError: If the elements in layers are not all positive integers.
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

            # Determine the size of the previous layer (or input size for first)
            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization for weights
            self.__weights['W{}'.format(i + 1)] = np.random.randn(
                layers[i], prev_size
            ) * np.sqrt(2 / prev_size)
            
            # Zero initialization for biases
            self.__weights['b{}'.format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Retrieves the number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        Retrieves the cache dictionary holding intermediary values.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Retrieves the weights dictionary holding weights and biases.
        """
        return self.__weights
