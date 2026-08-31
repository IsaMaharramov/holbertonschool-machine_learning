#!/usr/bin/env python3
"""
TF-IDF module
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    :param sentences: list of sentences to analyze
    :param vocab: list of the vocabulary words to use for the analysis
    :return: embeddings, features
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    
    try:
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        # Fallback for older versions of scikit-learn
        features = np.array(vectorizer.get_feature_names())
        
    return embeddings, features
