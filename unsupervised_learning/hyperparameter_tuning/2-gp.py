#!/usr/bin/env python3
"""
Module for Gaussian Process update.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor for GaussianProcess.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix
        using the Radial Basis Function (RBF).
        """
        sqdist = (
            np.sum(X1 ** 2, 1).reshape(-1, 1)
            + np.sum(X2 ** 2, 1)
            - 2 * np.dot(X1, X2.T)
        )
        return self.sigma_f ** 2 * np.exp(
            -0.5 / (self.l ** 2) * sqdist
        )

    def predict(self, X_s):
        """
        Predicts the mean and variance
        of points in a Gaussian process.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma_mat = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diagonal(sigma_mat)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process
        with new sample points.
        """
        self.X = np.vstack((self.X, X_new.reshape(1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(1, 1)))
        self.K = self.kernel(self.X, self.X)
