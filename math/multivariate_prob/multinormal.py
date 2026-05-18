#!/usr/bin/env python3
"""Module that defines the MultiNormal class."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initializes the Multivariate Normal distribution.

        Args:
            data (numpy.ndarray): Array of shape (d, n) containing the data.
                n is the number of data points.
                d is the number of dimensions in each data point.
                
        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        X_centered = data - self.mean
        self.cov = np.matmul(X_centered, X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Args:
            x (numpy.ndarray): Array of shape (d, 1) containing the data point.
                d is the number of dimensions of the Multinomial instance.

        Returns:
            float: The value of the PDF.
            
        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x is not of shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculate determinant and inverse of the covariance matrix
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)

        # Multivariate Normal distribution formula
        denominator = np.sqrt(((2 * np.pi) ** d) * det_cov)
        diff = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv_cov), diff)

        pdf_value = (1 / denominator) * np.exp(exponent[0, 0])

        return float(pdf_value)
