#!/usr/bin/env python3
"""
Module that defines a DeepNeuralNetwork with selectable activations.
"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """Initializes the deep neural network."""
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

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            prev_size = nx if i == 0 else layers[i - 1]
            self.__weights['W' + str(i + 1)] = np.random.randn(
                layers[i], prev_size) * np.sqrt(2 / prev_size)
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:
                exp_Z = np.exp(Z - np.max(Z, axis=0))
                A = exp_Z / np.sum(exp_Z, axis=0)
            else:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)
            self.__cache['A' + str(i)] = A
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates categorical cross-entropy cost."""
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """Evaluates the neural network."""
        A, _ = self.forward_prop(X)
        prediction = np.zeros(A.shape)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent."""
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W_curr = self.__weights['W' + str(i)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 1:
                if self.__activation == 'sig':
                    dZ = np.matmul(W_curr.T, dZ) * (cache['A' + str(i - 1)] * (1 - cache['A' + str(i - 1)]))
                else:
                    dZ = np.matmul(W_curr.T, dZ) * (1 - (cache['A' + str(i - 1)] ** 2))
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, step=100, graph=True, verbose=True):
        # ... validation logic ...
        
        for i in range(iterations + 1):
            if (i % step == 0 or i == iterations) and (graph or verbose):
                # Calculate cost once
                A, _ = self.forward_prop(X)
                cost = self.cost(Y, A)
                # Print only if verbose is requested or default
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            
            if i < iterations:
                _, cache = self.forward_prop(X)
                self.gradient_descent(Y, cache, alpha)
        
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
