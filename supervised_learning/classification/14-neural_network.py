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
            tuple: The private attributes __A1 and __A2.
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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels for the input data.

        Returns:
            tuple: The neuron's prediction and the cost.
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            A1 (numpy.ndarray): Output of the hidden layer.
            A2 (numpy.ndarray): Predicted output.
            alpha (float): The learning rate.
        """
        m = Y.shape[1]

        # Output layer backward propagation
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        # Keepdims=True ensures db2 is a matrix [[val]] rather than a scalar
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Hidden layer backward propagation
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels for the input data.
            iterations (int): Number of iterations.
            alpha (float): The learning rate.

        Returns:
            tuple: The evaluation of the training data after training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
