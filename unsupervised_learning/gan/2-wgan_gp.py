#!/usr/bin/env python3
""" Wasserstein GANs with gradient penalty """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class WGAN_GP(keras.Model):
    """ WGAN_GP class """
    
    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005, lambda_gp=10):
        """ Initialize the WGAN Gradient Penalty model """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.reduce_mean(x) - tf.reduce_mean(y)
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """ Generator of fake samples of size batch_size """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """ Generator of real samples of size batch_size """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """ Generator of interpolating samples of size batch_size """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """ Computing the gradient penalty """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)      

    def train_step(self, useless_argument):
        """ Overloading train_step() """
        for _ in range(self.disc_iter):
            # compute the penalized loss for the discriminator in a tape watching the discriminator's weights
            with tf.GradientTape() as disc_tape:
                # get a real sample
                real_sample = self.get_real_sample()
                # get a fake sample
                fake_sample = self.get_fake_sample(training=True)
                # get the interpolated sample (between real and fake computed above)
                interpolated_sample = self.get_interpolated_sample(real_sample, fake_sample)

                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)

                # compute the old loss discr_loss of the discriminator on real and fake samples
                discr_loss = self.discriminator.loss(fake_output, real_output)
                # compute the gradient penalty gp
                gp = self.gradient_penalty(interpolated_sample)
                # compute the sum new_discr_loss = discr_loss + self.lambda_gp * gp                     
                new_discr_loss = discr_loss + self.lambda_gp * gp
                
            # apply gradient descent with respect to new_discr_loss once to the discriminator 
            gradients_of_discriminator = disc_tape.gradient(new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # compute the loss for the generator in a tape watching the generator's weights 
        with tf.GradientTape() as gen_tape:
            # get a fake sample 
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_sample, training=False)
            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(fake_output)

        # apply gradient descent to the generator
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
