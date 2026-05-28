#!/usr/bin/env python3
"""Module to plot a scatter plot representing mountain elevation."""
import matplotlib.pyplot as plt
import numpy as np


def gradient():
    """Plots a 2D scatter field with an elevation color gradient."""
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # Plot scatter points using z values for color mapping
    scatter = plt.scatter(x, y, c=z, cmap='viridis')

    # Add labels and title matching specifications exactly
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    # Create and label the colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('elevation (m)')

    plt.show()
