#!/usr/bin/env python3
"""
Contains the Poisson class which represents a poisson distribution
"""


class Poisson:
    """
    Class that represents a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0

        lambtha = self.lambtha
        e = 2.7182818285

        # Factorial calculation: k!
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # PMF formula: (e^-lambda * lambda^k) / k!
        pmf_val = ((e ** -lambtha) * (lambtha ** k)) / factorial
        return pmf_val

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0

        # CDF is the sum of PMF values from 0 to k
        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)

        return cdf_val
