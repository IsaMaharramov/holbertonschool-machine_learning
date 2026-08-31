#!/usr/bin/env python3
"""
Extract Word2Vec module
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim word2vec model to a Keras Embedding layer.
    :param model: a trained gensim word2vec model
    :return: the trainable keras Embedding
    """
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors
    vocab_size, embedding_dim = weights.shape
    
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )
    
    return embedding_layer
