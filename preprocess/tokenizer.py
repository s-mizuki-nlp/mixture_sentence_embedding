#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from abc import ABCMeta, abstractmethod
from typing import List
from .wordpiece import FullTokenizer

class AbstractTokenizer(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def tokenize_single(self, sentence: str) -> List[str]:
        pass

    def tokenize(self, corpus):
        if (iter(corpus) is iter(corpus)) or isinstance(corpus, str):
            raise TypeError("`corpus` must be a container.")
        for sentence in corpus:
            yield self.tokenize_single(sentence)


class CharacterTokenizer(AbstractTokenizer):

    def __init__(self, separator: str=" ", do_lower_case = False):
        self._sep = separator
        self._do_lower_case = do_lower_case

    def tokenize_single(self, sentence):
        if self._do_lower_case:
            sentence = sentence.lower()
        return sentence.split(self._sep)


class WordPieceTokenizer(AbstractTokenizer):

    def __init__(self, vocab_file: str, do_lower_case = False):
        self._tokenizer = FullTokenizer(vocab_file, do_lower_case)

    def tokenize_single(self, sentence):
        return self._tokenizer.tokenize(sentence)