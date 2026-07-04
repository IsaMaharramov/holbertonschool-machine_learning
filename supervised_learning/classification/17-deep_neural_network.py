#!/usr/bin/env python3
"""
Module that defines a DeepNeuralNetwork for binary classification.
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

            # Determine the size of the previous layer (or nx for the first)
            if i == 0:
                prev_size = nx
            else:
                prev_size = layers[i - 1]

            # He et al. initialization for weights (using the missing *)
            self.__weights['W' + str(i + 1)] = np.random.randn(
                layers[i], prev_size) * np.sqrt(2 / prev_size)
            
            # Zero initialization for biases
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Getter method for the number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for the cache dictionary.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for the weights dictionary.
        """
        return self.__weights
