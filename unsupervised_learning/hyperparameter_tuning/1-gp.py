#!/usr/bin/env python3
"""
Module for Gaussian Process prediction.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor for GaussianProcess.

        Parameters:
        - X_init: numpy.ndarray of shape (t, 1), initial inputs.
        - Y_init: numpy.ndarray of shape (t, 1), initial outputs.
        - l: length parameter for the kernel.
        - sigma_f: standard deviation given to the output of the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).

        Parameters:
        - X1: numpy.ndarray of shape (m, 1)
        - X2: numpy.ndarray of shape (n, 1)

        Returns:
        - covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = (
            np.sum(X1 ** 2, 1).reshape(-1, 1) +
            np.sum(X2 ** 2, 1) -
            2 * np.dot(X1, X2.T)
        )
        return self.sigma_f ** 2 * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process.

        Parameters:
        - X_s: numpy.ndarray of shape (s, 1), points to predict.

        Returns:
        - mu: numpy.ndarray of shape (s,) containing the mean for each point.
        - sigma: numpy.ndarray of shape (s,) containing the variance for each point.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma_mat = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diagonal(sigma_mat)

        return mu, sigma
