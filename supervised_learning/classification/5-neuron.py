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

        Returns:
            The private attribute __A (the activated output).
        """
        Z = np.matmul(self.__W, X) + self.__b
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
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Args:
            X (numpy.ndarray): A numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy.ndarray): A numpy.ndarray with shape (1, m) that contains
                the correct labels for the input data.

        Returns:
            tuple: The neuron's prediction and the cost of the network,
            respectively.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): A numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy.ndarray): A numpy.ndarray with shape (1, m) that contains
                the correct labels for the input data.
            A (numpy.ndarray): A numpy.ndarray with shape (1, m) containing
                the activated output of the neuron for each example.
            alpha (float): The learning rate.
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db
