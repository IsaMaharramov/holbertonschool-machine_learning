#!/usr/bin/env python3
"""
Contains the Binomial class which represents a binomial distribution
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimation: p = 1 - (variance / mean)
            p_estimated = 1 - (variance / mean)
            self.n = int(round(mean / p_estimated))
            self.p = float(mean / self.n)

    def factorial(self, k):
        """
        Helper to calculate factorial
        """
        res = 1
        for i in range(1, k + 1):
            res *= i
        return res

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # nCr = n! / (k!(n-k)!)
        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        nk_fact = self.factorial(self.n - k)
        combination = n_fact / (k_fact * nk_fact)

        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)
        return cdf_val
