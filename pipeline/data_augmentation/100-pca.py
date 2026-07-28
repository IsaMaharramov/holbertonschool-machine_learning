#!/usr/bin/env python3
"""Module containing the pca_color function for PCA color augmentation."""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: a 3D tf.Tensor containing the image to change.
        alphas: a tuple/list of length 3 containing the alpha values (amount
                that each channel should change).

    Returns:
        The PCA color augmented image tensor (uint8 or original type).
    """
    # Preserve original dtype to cast back before returning
    orig_dtype = image.dtype

    # Convert image to float32 for mathematical calculations
    img_float = tf.cast(image, tf.float32)

    # Reshape image to (N, 3) where N = height * width
    pixels = tf.reshape(img_float, [-1, 3])

    # Mean center the pixels along the channel axis
    mean = tf.reduce_mean(pixels, axis=0)
    centered_pixels = pixels - mean

    # Compute covariance matrix: (3, 3)
    # Covariance = (1 / (N - 1)) * (X^T @ X)
    num_pixels = tf.cast(tf.shape(pixels)[0], tf.float32)
    cov = tf.matmul(centered_pixels, centered_pixels, transpose_a=True)
    cov = cov / (num_pixels - 1.0)

    # Compute eigenvalues (3,) and eigenvectors (3, 3)
    e_val, e_vec = tf.linalg.eigh(cov)

    # Cast alphas to float32 tensor of shape (3,)
    alpha_tensor = tf.cast(alphas, tf.float32)

    # Calculate perturbation vector: p_i * alpha_i * lambda_i
    # e_vec: columns are eigenvectors [p1, p2, p3]
    # alpha_tensor * e_val: (3,)
    delta = tf.matmul(e_vec, tf.reshape(alpha_tensor * e_val, (3, 1)))
    delta = tf.reshape(delta, (1, 1, 3))

    # Add perturbation to original float image and clip to valid [0, 255] range
    augmented_img = img_float + delta
    augmented_img = tf.clip_by_value(augmented_img, 0.0, 255.0)

    return tf.cast(augmented_img, orig_dtype)
