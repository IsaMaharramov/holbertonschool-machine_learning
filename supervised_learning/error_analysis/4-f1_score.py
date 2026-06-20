#!/usr/bin/env python3
"""
Module for calculating F1 Score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.

    parameters:
        confusion [numpy.ndarray of shape (classes, classes)]:
            row indices represent the correct labels and column indices
            represent the predicted labels
            classes is the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score
        of each class.
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * sens) / (prec + sens)
