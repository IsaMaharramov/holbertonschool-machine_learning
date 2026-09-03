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

    def __init__(self, batch_size, max_len):
        """
        Class constructor

        Args:
            batch_size: batch size for training/validation
            max_len: maximum number of tokens allowed per example sentence
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(pt, en):
            """Filters out examples longer than max_len"""
            return tf.logical_and(tf.size(pt) <= max_len,
                                  tf.size(en) <= max_len)

        # Update train dataset pipeline
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Update validation dataset pipeline
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size)

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

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens, en_tokens
        """
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
