#!/usr/bin/env python3
"""
Module to plot a scatter graph
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots height vs weight as a magenta scatter plot
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.scatter(x, y, color='magenta')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()
