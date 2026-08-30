#!/usr/bin/env python3
"""
Variational Autoencoder module.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Parameters:
    - input_dims: integer containing the dimensions of the model input
    - hidden_layers: list containing nodes -> each hidden layer in encoder
    - latent_dims: integer containing the dimensions of the latent space

    Returns:
    - encoder: encoder model
    - decoder: decoder model
    - auto: full autoencoder model
    """
    # Encoder
    X_input = keras.Input(shape=(input_dims,))
    Y_prev = X_input
    for h in hidden_layers:
        Y_prev = keras.layers.Dense(units=h, activation='relu')(Y_prev)

    z_mean = keras.layers.Dense(units=latent_dims, activation=None)(Y_prev)
    z_log_sigma = keras.layers.Dense(units=latent_dims, activation=None)(Y_prev)

    def sampling(args):
        """Sampling similar points in latent space."""
        z_m, z_sig = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.int_shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(z_sig / 2) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,)
    )([z_mean, z_log_sigma])

    encoder = keras.Model(X_input, [z, z_mean, z_log_sigma], name='encoder')

    # Decoder
    X_decode = keras.Input(shape=(latent_dims,))
    Y_prev = X_decode
    for h in reversed(hidden_layers):
        Y_prev = keras.layers.Dense(units=h, activation='relu')(Y_prev)

    output = keras.layers.Dense(units=input_dims, activation='sigmoid')(Y_prev)
    decoder = keras.Model(X_decode, output, name='decoder')

    # Full Autoencoder (passing sampled z, which is index 0)
    outputs = decoder(encoder(X_input)[0])
    auto = keras.Model(X_input, outputs, name='autoencoder')

    def vae_loss(x, x_decoder_mean):
        """Computes reconstruction loss + KL divergence loss."""
        x_loss = keras.backend.binary_crossentropy(x, x_decoder_mean)
        x_loss = keras.backend.sum(x_loss, axis=1)
        kl_loss = -0.5 * keras.backend.mean(
            1 + z_log_sigma - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_sigma), axis=-1
        )
        return x_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
