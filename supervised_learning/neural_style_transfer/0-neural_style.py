#!/usr/bin/env python3
"""
Neural Style Transfer Module
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer class that performs tasks for neural style transfer.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for the NST class.
        
        Args:
            style_image (np.ndarray): The image used as a style reference.
            content_image (np.ndarray): The image used as a content reference.
            alpha (float): The weight for content cost.
            beta (float): The weight for style cost.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.
        
        Args:
            image (np.ndarray): The image to be scaled.
            
        Returns:
            tf.Tensor: The scaled image with shape (1, h_new, w_new, 3).
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        
        h, w, _ = image.shape
        scale = 512 / max(h, w)
        h_new = int(h * scale)
        w_new = int(w * scale)
        
        # Add batch dimension to the image
        image_expanded = tf.expand_dims(image, axis=0)
        
        # Resize using bicubic interpolation
        image_resized = tf.image.resize(image_expanded, size=(h_new, w_new),
                                        method='bicubic')
        
        # Rescale the pixel values from [0, 255] to [0, 1]
        image_rescaled = image_resized / 255.0
        
        # Clip values to ensure they stay strictly within [0, 1] 
        # (bicubic interpolation can result in values slightly outside this range)
        image_scaled = tf.clip_by_value(image_rescaled, 0.0, 1.0)
        
        return image_scaled
