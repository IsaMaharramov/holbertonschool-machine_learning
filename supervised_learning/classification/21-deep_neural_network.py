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
        """Initializes the deep neural network."""
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

            prev_size = nx if i == 0 else layers[i - 1]

            self.__weights['W' + str(i + 1)] = np.random.randn(
                layers[i], prev_size) * np.sqrt(2 / prev_size)
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter -> the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter -> the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter -> the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i)] = A

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): shape (1, m) containing correct labels.
            cache (dict): dictionary holding intermediary values of the network.
            alpha (float): the learning rate.
        """
        m = Y.shape[1]
        # Calculate dZ -> the final layer
        dZ = cache['A' + str(self.__L)] - Y

        # Iterate backward from the final layer to the first
        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            # Get current weights before updating them to calculate next dZ
            W_curr = self.__weights['W' + str(i)]

            # Calculate gradients
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Calculate dZ -> the previous layer (if not at layer 0)
            if i > 1:
                # Derivative of sigmoid: g'(z) = A * (1 - A)
                dg = A_prev * (1 - A_prev)
                dZ = np.matmul(W_curr.T, dZ) * dg

            # Update weights and biases
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db
