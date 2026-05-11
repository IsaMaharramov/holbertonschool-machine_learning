#!/usr/bin/env python3
"""
Contains the continuous posterior function for Bayesian probability
"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability
    of developing severe side effects falls within a specific range
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # The posterior of a Binomial likelihood with a Uniform prior
    # is a Beta distribution: Beta(alpha, beta)
    # alpha = x + 1, beta = n - x + 1
    alpha = x + 1
    beta = n - x + 1

    # The probability p is in [p1, p2] is CDF(p2) - CDF(p1)
    # scipy.special.betainc is the regularized incomplete beta function
    # which acts as the CDF for the Beta distribution
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    return cdf_p2 - cdf_p1
