#!/usr/bin/env python3
"""This module defines the ngram_bleu function."""
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


def ngram_bleu(references, sentence, n):
    """
    This function calculates the n-gram BLEU score for a sentence.
    Args:
        references (list of list of str): is a list of reference translations.
            Each reference is a list of the words in the translation.
        sentence (list of str): is a list containing the model proposed
            sentence.
        n (int): The size of the n-gram to use for evaluation.
    Returns:
        float: The n-gram BLEU score.
    """
    # Generate n-grams for the candidate sentence
    candidate_ngrams = generate_ngrams(sentence, n)
    candidate_counts = Counter(candidate_ngrams)

    # Generate n-grams for each reference
    reference_ngrams = [generate_ngrams(ref, n) for ref in references]

    # Count clipped n-grams
    clipped_count = 0
    total_ngrams = len(candidate_ngrams)

    for ngram in candidate_counts:
        max_ref_count = max(ref.count(ngram) for ref in reference_ngrams)
        clipped_count += min(candidate_counts[ngram], max_ref_count)

    # Calculate modified precision
    precision = clipped_count / total_ngrams if total_ngrams > 0 else 0

    # Calculate the length of the candidate and the best reference
    c = len(sentence)
    r = min(references, key=lambda ref: abs(len(ref) - c))
    r = len(r)

    # Compute Brevity Penalty (BP)
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)

    # Final BLEU score
    bleu_score = BP * precision

    return bleu_score
