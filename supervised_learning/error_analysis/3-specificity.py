#!/usr/bin/env python3
"""
Module for calculating specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    parameters:
        confusion [numpy.ndarray of shape (classes, classes)]:
            row indices represent the correct labels and column indices
            represent the predicted labels
            classes is the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    actual_positives = np.sum(confusion, axis=1)
    total_population = np.sum(confusion)

    true_negatives = total_population - actual_positives - predicted_positives + true_positives
    false_positives = predicted_positives - true_positives
    
    return true_negatives / (true_negatives + false_positives)
