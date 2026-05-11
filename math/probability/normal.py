#!/usr/bin/env python3
"""
Contains the Normal class which represents a normal distribution
"""


class Normal:
    """
    Class that represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            # Calculate variance to find standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        exponent = -0.5 * (self.z_score(x) ** 2)
        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        return coefficient * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536
        # Argument for the erf function
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Broken down Taylor series for erf to satisfy E501
        erf_series = (z - (z ** 3) / 3 + (z ** 5) / 10 -
                      (z ** 7) / 42 + (z ** 9) / 216)
        erf = (2 / (pi ** 0.5)) * erf_series

        return 0.5 * (1 + erf)