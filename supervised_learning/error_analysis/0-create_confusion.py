#!/usr/bin/env python3
"""
Module for creating a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    parameters:
        labels [numpy.ndarray of shape (m, classes)]:
            contains the correct labels for each data point
            m is the number of data points
            classes is the number of classes
        logits [numpy.ndarray of shape (m, classes)]:
            contains the predicted labels

    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing
        the predicted labels.
    """
    return np.matmul(labels.T, logits)
