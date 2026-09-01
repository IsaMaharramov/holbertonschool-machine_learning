#!/usr/bin/env python3
"""This module defines the cumulative_bleu function."""
from collections import Counter
import numpy as np


def generate_ngrams(sentence, n):
    """
    This function generates n-grams from a sentence.
    Args:
        sentence (list of str): The input sentence (list of words).
        n (int): The size of the n-grams to generate.
    Returns:
        list of str: A list of n-grams.
    """
    return [" ".join(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]


def cumulative_bleu(references, sentence, n):
    """
    This function calculates the cumulative n-gram BLEU score for a sentence.
    Args:
        rreferences (list of list of str): is a list of reference translations.
            Each reference is a list of the words in the translation.
        sentence (list of str): is a list containing the model proposed
            sentence.
        n (int): The size of the largest n-gram to use for evaluation.
    Returns:
        float: The cumulative n-gram BLEU score.
    """
    precisions = []

    for k in range(1, n + 1):
        # Generate k-grams for the candidate sentence
        candidate_ngrams = generate_ngrams(sentence, k)
        candidate_counts = Counter(candidate_ngrams)

        # Generate k-grams for each reference
        reference_ngrams = [generate_ngrams(ref, k) for ref in references]

        # Count clipped k-grams
        clipped_count = 0
        total_ngrams = len(candidate_ngrams)

        for ngram in candidate_counts:
            max_ref_count = max(ref.count(ngram) for ref in reference_ngrams)
            clipped_count += min(candidate_counts[ngram], max_ref_count)

        # Calculate modified precision
        if total_ngrams > 0:
            precisions.append(clipped_count / total_ngrams)
        else:
            precisions.append(0)

    # Calculate the length of the candidate and the best reference
    c = len(sentence)
    r = min(references, key=lambda ref: abs(len(ref) - c))
    r = len(r)

    # Compute Brevity Penalty (BP)
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)

    # Compute the cumulative BLEU score
    if all(precision > 0 for precision in precisions):
        score = BP * np.exp(np.mean(np.log(precisions)))
    else:
        score = 0

    return score
