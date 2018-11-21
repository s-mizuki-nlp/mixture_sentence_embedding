#!/usr/bin/env python
# -*- coding:utf-8 -*-

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from typing import Union, Optional, List, Dict
from copy import deepcopy
from collections import Counter
import warnings
import pickle
from gensim.models import Word2Vec
import numpy as np

class Dictionary(object):

    __oov = "__oov__"

    def __init__(self, special_tokens: Optional[List[str]] = None, masking: bool=True, oov: bool=False, count_freq: bool=False):

        self._token2id = {}
        self._id2token = {}
        self._masking = masking
        if special_tokens is not None:
            self._offset = len(special_tokens) + masking
        else:
            self._offset = int(masking)
        self._count_freq = count_freq
        self._oov = oov
        self._special_tokens = deepcopy(special_tokens) if special_tokens is not None else []
        if self._oov:
            self._special_tokens.append(self.__oov)
        self._oov_id = None
        self._init()

    def _init(self):

        self._token2id = {}
        self._id2token = {}
        self._counter = Counter()

        idx = int(self._masking)
        for token in self._special_tokens:
            self._token2id[token] = idx
            self._id2token[idx] = token
            idx += 1

        self._oov_id = self._token2id[self.__oov] if self._oov else None

    def _compactify(self):

        # reset internal state
        token2id_old = deepcopy(self._token2id)
        counter_old = deepcopy(self._counter)
        self._init()
        self._counter = counter_old

        idx = int(self._masking)
        for token, idx_old in sorted(token2id_old.items(), key = lambda pair: pair[-1], reverse=False):
            self._token2id[token] = idx
            self._id2token[idx] = token
            idx += 1

    @property
    def n_vocab(self) -> int:
        return len(self._token2id)

    @property
    def max_id(self) -> int:
        if len(self._id2token) > 0:
            return max(self._id2token.keys())
        else:
            return self._offset

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def masking(self) -> bool:
        return self._masking

    @property
    def oov_id(self) -> Optional[int]:
        return self._oov_id

    @property
    def special_tokens(self) -> Dict[str, int]:
        return {token:self._token_to_id(token) for token in self._special_tokens}

    @property
    def vocab(self):
        return self._token2id.keys()

    def save(self, file_path: str):
        if len(self._token2id) == 0:
            warnings.warn("dictionary is empty. did you call `fit()` method?")
        with io.open(file_path, mode="wb") as ofs:
            pickle.dump(self, ofs)

    @classmethod
    def load(cls, file_path: str):
        with io.open(file_path, mode="rb") as ifs:
            obj = pickle.load(ifs)
        obj.__class__ = cls
        return obj

    def clear(self):
        self._init()

    def fit(self, tokenized_corpus, initialize=True):
        if initialize:
            self._init()
        idx = self.max_id + 1
        for lst_token in tokenized_corpus:
            for token in lst_token:
                if token not in self._token2id:
                    self._token2id[token] = idx
                    self._id2token[idx] = token
                    idx += 1
            if self._count_freq:
                self._counter.update(lst_token)

    def filter_extremes(self, keep_n: Optional[int] = None, no_below: Optional[int] = None):
        """
        filter out extreme tokens. equivalent to the gensim.corpora.Dictionary.filter_extremes() method.

        :param keep_n: keeps tokens within top-n occurence
        :param no_below: remove tokens below specified occurence
        :return:
        """
        if not self._count_freq:
            warnings.warn("you must enable `count_freq` to use this feature.")
            return

        if keep_n is None and no_below is None:
            warnings.warn("you must specify either `keep_n` or `no_below` argument.")
            return

        n_vocab_before = self.n_vocab

        if no_below is not None:
            counter_old = deepcopy(self._counter)
            for s_token in self._special_tokens:
                counter_old.pop(s_token, None)
            for token, freq in counter_old.items():
                if freq < no_below:
                    del self._token2id[token]
                    del self._counter[token]

        if keep_n is not None:
            n_diff = self.n_vocab - keep_n
            if n_diff > 0:
                counter_old = deepcopy(self._counter)
                for s_token in self._special_tokens:
                    counter_old.pop(s_token, None)
                for token in sorted(counter_old, key=counter_old.get, reverse=False)[:n_diff]:
                    del self._token2id[token]
                    del self._counter[token]

        if self.n_vocab != n_vocab_before:
            self._compactify()

    def token(self, token: str) -> (int, int):
        """
        if exists, returns token id and its frequency
        :param token: string
        :return: (token_id, frequency)
        """
        return self._token_to_id(token), self._counter.get(token, 0)

    def _token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self._oov_id)

    def _id_to_token(self, index: int) -> str:
        return self._id2token.get(index, None)

    def __getitem__(self, item: str) -> int:
        return self._token_to_id(item)

    def transform(self, lst_token):
        return [self._token_to_id(token) for token in lst_token]

    def iter_transform(self, iter_lst_token):
        for lst_token in iter_lst_token:
            yield self.transform(lst_token)

    def inverse_transform(self, lst_index):
        return [self._id_to_token(index) for index in lst_index]

    def iter_inverse_transform(self, iter_lst_index):
        for lst_index in iter_lst_index:
            yield self.inverse_transform(lst_index)

    # ToDo: add fastText model support
    def to_word_embedding(self, model_w2v: Word2Vec, dtype=np.float32):

        n_dim = model_w2v.vector_size
        mat_ret = np.zeros((self.n_vocab + self.masking, model_w2v.vector_size), dtype=dtype)

        def random_vector():
            ret = np.random.normal(size=n_dim).astype(dtype)
            ret /= np.linalg.norm(ret)
            return ret

        for idx, token in self._id2token.items():
            if token in model_w2v.wv:
                mat_ret[idx] = model_w2v.wv[token]
            else:
                mat_ret[idx] = random_vector()

        return mat_ret