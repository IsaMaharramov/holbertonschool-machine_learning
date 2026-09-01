#!/usr/bin/env python3
""" This module defines the uni_bleu function """
import numpy as np


def uni_bleu(references, sentence):
    """
    This function calculates the unigram BLEU score for a sentence.
    Args:
        references (list of list of str): is a list of reference translations.
            Each reference is a list of the words in the translation.
        sentence (list of str): is a list containing the model proposed
            sentence.
    Returns:
        float: The unigram BLEU score.
    """
    # Count unigram occurrences in the candidate sentence
    candidate_counts = {}
    for word in sentence:
        candidate_counts[word] = candidate_counts.get(word, 0) + 1

    # Count clipped unigrams (maximum count allowed by any reference)
    clipped_count = 0
    total_unigrams = len(sentence)

    for word in candidate_counts:
        max_ref_count = max(ref.count(word) for ref in references)
        clipped_count += min(candidate_counts[word], max_ref_count)

    # Calculate modified precision
    precision = clipped_count / total_unigrams

    # Calculate the length of the candidate and the best reference
    c = len(sentence)
    # Closest reference length
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
