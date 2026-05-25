#!/usr/bin/env python3
"""
Module to plot a simple line graph
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a solid red line representing a cubic function
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()
