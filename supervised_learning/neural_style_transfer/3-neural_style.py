#!/usr/bin/env python3
"""
Module for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class NST that performs tasks for neural style transfer.
    """
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor for the NST class.

        Args:
            style_image: image used as a style reference (numpy.ndarray)
            content_image: image used as a content reference (numpy.ndarray)
            alpha: weight for content cost (non-negative number)
            beta: weight for style cost (non-negative number)
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

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image: A numpy.ndarray of shape (h, w, 3) containing the image
                   to be scaled.

        Returns:
            The scaled image as a tf.tensor with shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or \
           image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape

        # Calculate new dimensions, keeping aspect ratio and max side = 512
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        # Convert to tensor and expand dims to (1, h, w, 3)
        image_tf = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tf = tf.expand_dims(image_tf, axis=0)

        # Resize using bicubic interpolation
        image_tf = tf.image.resize(image_tf,
                                   size=[h_new, w_new],
                                   method='bicubic')

        # Rescale pixel values from [0, 255] to [0, 1] and clip to bounds
        image_tf = image_tf / 255.0
        image_tf = tf.clip_by_value(image_tf, 0.0, 1.0)

        return image_tf

    def load_model(self):
        """
        Creates the model used to calculate cost.
        Loads VGG19, replaces MaxPooling with AveragePooling, and updates
        the model instance to fetch features for style and content extraction.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        def clone_function(layer):
            """
            Replaces MaxPooling2D with AveragePooling2D.
            """
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                return tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
            return layer

        # Clone model to preserve identical layer names and weights structures
        custom_vgg = tf.keras.models.clone_model(vgg,
                                                 clone_function=clone_function)
        custom_vgg.set_weights(vgg.get_weights())

        # Grab specific layer outputs for style and content extraction
        outputs = [custom_vgg.get_layer(name).output
                   for name in self.style_layers + [self.content_layer]]

        # Construct the final model
        self.model = tf.keras.Model(inputs=custom_vgg.input, outputs=outputs)

        # The VGG parameters should freeze and not be updated during training
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a given layer output.

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)
                         containing the layer output.

        Returns:
            tf.Tensor of shape (1, c, c) containing the gram matrix.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # The gram matrix G can be computed efficiently using einsum.
        # 'bijc,bijd->bcd' means we multiply and sum over height (i) and
        # width (j) dimensions, keeping batch (b) and channels (c and d).
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Obtain the dimensions to normalize the matrix
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        # Normalize the Gram matrix by the number of locations (h * w)
        return result / num_locations

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        Sets the public instance attributes:
            gram_style_features: list of gram matrices from style layers
            content_feature: content layer output from the content image
        """
        # Pass the preprocessed images through the model
        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)

        # Calculate Gram matrices for the style layer outputs
        # Style layers are all outputs except the last one
        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_outputs[:-1]
        ]

        # Content layer is the last output
        self.content_feature = content_outputs[-1]
