#!/usr/bin/env python3
"""Module for PCA color augmentation."""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: a 3D tf.Tensor containing the image to change.
        alphas: a tuple of length 3 containing the amount each channel
                should change.

    Returns:
        The augmented image tensor.
    """
    orig_dtype = image.dtype
    img_float = tf.cast(image, tf.float32)

    # Reshape image pixels to (N, 3)
    reshaped_img = tf.reshape(img_float, [-1, 3])

    # Mean subtract across channels
    mean = tf.reduce_mean(reshaped_img, axis=0)
    centered = reshaped_img - mean

    # Compute 3x3 covariance matrix over channels
    num_pixels = tf.cast(tf.shape(reshaped_img)[0], tf.float32)
    cov = tf.matmul(centered, centered, transpose_a=True) / num_pixels

    # Compute eigenvalues and eigenvectors
    e_val, e_vec = tf.linalg.eigh(cov)

    # Format alphas into tensor shape (3, 1)
    alphas_tensor = tf.cast(alphas, tf.float32)
    alphas_tensor = tf.reshape(alphas_tensor, (3, 1))

    # e_val needs to be (3, 1) to multiply element-wise with alphas
    e_val = tf.reshape(e_val, (3, 1))

    # Linear combination: p_i * alpha_i * lambda_i
    # e_vec @ (alphas * e_val)
    delta = tf.matmul(e_vec, alphas_tensor * e_val)
    delta = tf.reshape(delta, (1, 1, 3))

    # Add perturbation, clip to valid RGB pixel range [0, 255]
    augmented = img_float + delta
    augmented = tf.clip_by_value(augmented, 0.0, 255.0)

    return tf.cast(augmented, orig_dtype)
