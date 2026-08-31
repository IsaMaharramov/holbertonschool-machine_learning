#!/usr/bin/env python3
"""
Bag of Words Module
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list): List of the vocabulary words to use for the analysis.
            If None, all words within sentences should be used.

    Returns:
        embeddings (numpy.ndarray): Embeddings matrix of shape (s, f).
        features (list): List of features used for embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()

    try:
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        features = np.array(vectorizer.get_feature_names())

    return embeddings, features
