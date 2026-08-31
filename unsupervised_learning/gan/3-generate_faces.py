#!/usr/bin/env python3
""" Generating faces using CNNs """
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """ Builds a generator and discriminator CNN models """

    def get_generator():
        """ Generator model """
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048)(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """ Discriminator model """
        inputs = keras.Input(shape=(16, 16, 1))

        x = keras.layers.Conv2D(32, kernel_size=3, padding="same")(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Conv2D(128, kernel_size=3, padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Conv2D(256, kernel_size=3, padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1)(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
