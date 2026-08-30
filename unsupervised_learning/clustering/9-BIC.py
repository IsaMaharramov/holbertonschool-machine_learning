#!/usr/bin/env python3
"""
Module -> Bayesian Information Criterion (BIC) calculation in GMMs.
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters -> a GMM using the Bayesian
    Information Criterion.

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing the minimum number of clusters
    - kmax: positive integer containing the maximum number of clusters
    - iterations: positive integer containing max iterations -> EM algorithm
    - tol: non-negative float containing tolerance -> EM algorithm
    - verbose: boolean determining if EM algorithm should print info

    Returns:
    - best_k: best value -> k based on its BIC
    - best_result: tuple containing (pi, m, S) -> the best k
    - l: numpy.ndarray containing log likelihood -> each cluster size tested
    - b: numpy.ndarray containing BIC value -> each cluster size tested
    - Returns None, None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None, None, None
    if kmin > kmax or kmax > X.shape[0]:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float and type(tol) is not int:
        return None, None, None, None
    if tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    results = {}
    logl_val = []
    bic_val = []
    n, d = X.shape

    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_l = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        if pi is None:
            return None, None, None, None
        logl_val.append(log_l)

        # Number of parameters -> GMM with full covariance matrices
        cov_params = k * d * (d + 1) / 2.0
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        # BIC formula: BIC = p * ln(n) - 2 * l
        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)
        results[k] = (pi, m, S)

    logl_val = np.array(logl_val)
    bic_val = np.array(bic_val)

    best_idx = np.argmin(bic_val)
    best_k = list(results.keys())[best_idx]
    best_result = results[best_k]

    return best_k, best_result, logl_val, bic_val
