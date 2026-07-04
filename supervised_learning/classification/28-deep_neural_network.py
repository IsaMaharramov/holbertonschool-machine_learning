#!/usr/bin/env python3
"""
Module that defines a DeepNeuralNetwork with selectable activations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initializes the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer of the network.
            activation (str): Type of activation function ('sig' or 'tanh').
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization
            self.__weights["W" + str(i + 1)] = np.random.randn(
                layers[i], prev_size) * np.sqrt(2 / prev_size)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Returns the number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Returns the intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Returns the weights and biases of the network."""
        return self.__weights

    @property
    def activation(self):
        """Returns the activation function type used in hidden layers."""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            A_prev = self.__cache["A" + str(i - 1)]

            Z = np.matmul(W, A_prev) + b

            if i == self.L:
                # Softmax activation for the output layer
                t = np.exp(Z - np.max(Z, axis=0))
                A = t / np.sum(t, axis=0)
            else:
                # Selectable activation for hidden layers
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    A = np.tanh(Z)

            self.__cache["A" + str(i)] = A

        return self.__cache["A" + str(self.L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the categorical cross-entropy cost of the model.
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        # Convert output probabilities into a one-hot encoded matrix
        prediction = np.where(A == np.max(A, axis=0), 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
        """
        m = Y.shape[1]
        dZ = cache["A" + str(self.L)] - Y

        for i in range(self.L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            W = self.__weights["W" + str(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 1:
                A_curr = cache["A" + str(i - 1)]
                if self.__activation == 'sig':
                    dZ = np.matmul(W.T, dZ) * (A_curr * (1 - A_curr))
                elif self.__activation == 'tanh':
                    dZ = np.matmul(W.T, dZ) * (1 - (A_curr ** 2))

            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
