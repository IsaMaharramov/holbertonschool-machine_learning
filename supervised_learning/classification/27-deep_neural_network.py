#!/usr/bin/env python3
"""
Module that defines a DeepNeuralNetwork -> multiclass classification.
"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification."""
    
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
    def L(self): return self.__L
    @property
    def cache(self): return self.__cache
    @property
    def weights(self): return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            
            if i == self.__L:
                # Softmax -> multiclass
                exp_Z = np.exp(Z - np.max(Z, axis=0))
                A = exp_Z / np.sum(exp_Z, axis=0)
            else:
                # Sigmoid -> hidden layers
                A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i)] = A
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using categorical cross-entropy."""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.zeros(A.shape)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, cost

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
                A = cache['A' + str(i - 1)]
                dZ = np.matmul(W_curr.T, dZ) * (A * (1 - A))
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, step=100, graph=True):
        """Trains the deep neural network."""
        # ... (Include validation -> iterations, alpha) ...
        # ... (Loop logic as before) ...
        pass # Note: Ensure logic follows requirements -> step/graph

    def save(self, filename):
        """Saves the instance object."""
        if not filename.endswith('.pkl'): filename += '.pkl'
        with open(filename, 'wb') as f: pickle.dump(self, f)

    @staticmethod

    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        if not os.path.exists(filename): return None
        with open(filename, 'rb') as f: return pickle.load(f)
