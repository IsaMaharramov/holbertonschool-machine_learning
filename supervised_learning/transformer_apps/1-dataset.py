#!/usr/bin/env python3
"""
Dataset module for Transformer Applications
"""
import tensorflow as tf
from transformers import AutoTokenizer
from setup import load_pt2en


class Dataset:
    """
    Dataset class that loads and preps a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset

        Args:
            data: a tf.data.Dataset whose examples are formatted as
                  a tuple (pt, en)

        Returns:
            tokenizer_pt, tokenizer_en
        """
        pt_tokenizer_base = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        en_tokenizer_base = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            """Iterator for the Portuguese dataset"""
            for pt, en in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_iterator():
            """Iterator for the English dataset"""
            for pt, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = pt_tokenizer_base.train_new_from_iterator(
            pt_iterator(), 2**13
        )
        tokenizer_en = en_tokenizer_base.train_new_from_iterator(
            en_iterator(), 2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens, en_tokens
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        pt_tokens = [pt_vocab_size] + \
            self.tokenizer_pt.encode(pt_str, add_special_tokens=False) + \
            [pt_vocab_size + 1]

        en_tokens = [en_vocab_size] + \
            self.tokenizer_en.encode(en_str, add_special_tokens=False) + \
            [en_vocab_size + 1]

        return pt_tokens, en_tokens
