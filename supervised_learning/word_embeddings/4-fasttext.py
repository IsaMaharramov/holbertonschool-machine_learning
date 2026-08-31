#!/usr/bin/env python3
"""
FastText module
"""
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model
    :param sentences: list of sentences to be trained on
    :param vector_size: dimensionality of the embedding layer
    :param min_count: minimum number of occurrences of a word for use in training
    :param window: maximum distance between the current and predicted word
    :param negative: size of negative sampling
    :param cbow: boolean to determine the training type; True is CBOW, False is Skip-gram
    :param epochs: number of iterations to train over
    :param seed: seed for the random number generator
    :param workers: number of worker threads to train the model
    :return: the trained model
    """
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
