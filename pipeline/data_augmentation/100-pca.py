#!/usr/bin/env python3
"""Module for PCA color augmentation."""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: a 3D tf.Tensor containing the image to change.
        alphas: a tuple of length 3 containing the amount that each channel
                should change.

    Returns:
        The augmented image tensor.
    """
    # Preserve original dtype to cast back later
    orig_dtype = image.dtype
    
    # Cast the image to float32 for calculations
    img = tf.cast(image, tf.float32)

    # Flatten the image to 2D array (num_pixels, 3)
    flat_img = tf.reshape(img, [-1, 3])

    # Mean center the pixels across the channel axis
    mean = tf.reduce_mean(flat_img, axis=0)
    centered = flat_img - mean

    # Compute the 3x3 covariance matrix using TensorFlow
    # Normalized by N-1
    num_pixels = tf.cast(tf.shape(centered)[0] - 1, tf.float32)
    cov = tf.tensordot(tf.transpose(centered), centered, axes=1) / num_pixels

    # Perform eigendecomposition using TensorFlow
    eigen_values, eigen_vectors = tf.linalg.eigh(cov)

    # Convert the alphas tuple to a float32 tensor
    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)

    # Compute the perturbation delta
    # Broadcast element-wise multiplication on eigenvectors, sum across axis 1
    # This mimics [p1, p2, p3] @ [a1*L1, a2*L2, a3*L3]^T purely in TF
    delta = tf.reduce_sum(eigen_vectors * (alphas * eigen_values), axis=1)

    # Add the perturbation vector directly to the image
    pca_img = img + delta

    # Clip the pixel values to the valid [0, 255] range
    pca_img = tf.clip_by_value(pca_img, 0.0, 255.0)

    # Explicitly cast back to original data type (e.g., uint8)
    return tf.cast(pca_img, orig_dtype)
