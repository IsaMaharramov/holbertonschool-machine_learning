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
        self.load_model()

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
        
        image_expanded = tf.expand_dims(image, axis=0)
        
        image_resized = tf.image.resize(image_expanded, size=(h_new, w_new),
                                        method='bicubic')
        
        image_rescaled = image_resized / 255.0
        image_scaled = tf.clip_by_value(image_rescaled, 0.0, 1.0)
        
        return image_scaled

    def load_model(self):
        """
        Creates the model used to calculate cost for Neural Style Transfer.
        Uses VGG19 as the base model, replaces MaxPooling with AveragePooling,
        and saves it to the instance attribute `model`.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        
        x = vgg.input
        outputs = {}
        
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
                
            if layer.name in self.style_layers + [self.content_layer]:
                outputs[layer.name] = x
                
            if layer.name == self.content_layer:
                break
                
        output_tensors = [outputs[name] for name in 
                          self.style_layers + [self.content_layer]]
        
        self.model = tf.keras.Model(inputs=vgg.input, outputs=output_tensors)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer tensor.
        
        Args:
            input_layer (tf.Tensor or tf.Variable): Tensor of shape (1, h, w, c).
            
        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c).
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError(
                "input_layer must be a tensor of rank 4")
        
        # Reshape to (1, h * w, c) to prepare for batch matrix multiplication
        shape = tf.shape(input_layer)
        batch, h, w, c = shape[0], shape[1], shape[2], shape[3]
        
        features = tf.reshape(input_layer, (batch, h * w, c))
        
        # Compute the Gram matrix via batched matrix multiplication: (1, c, h*w) x (1, h*w, c) -> (1, c, c)
        gram = tf.matmul(features, features, transpose_a=True)
        
        # Normalize by the number of locations (h * w)
        num_locations = tf.cast(h * w, tf.float32)
        gram = gram / num_locations
        
        return gram
