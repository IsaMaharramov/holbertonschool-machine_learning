#!/usr/bin/env python3
"""
Visualizes high dimensional data (Iris dataset) in 3D using 
Principal Component Analysis (PCA).
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

# Perform PCA to reduce dimensionality
data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Initialize the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter using pca_data columns for x, y, and z respectively
ax.scatter(pca_data[:, 0], 
           pca_data[:, 1], 
           pca_data[:, 2], 
           c=labels, 
           cmap='plasma')

# Set the title and axis labels
ax.set_title("PCA of Iris Dataset")
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")

# Display the plot
plt.show()
