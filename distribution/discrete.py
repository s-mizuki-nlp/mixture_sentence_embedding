#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

class CategoricalDistributionUsingAliasMethod(object):

    def __init__(self, p: np.ndarray):

        assert np.abs(1. - np.sum(p)) < 1E-6, "sum of probability must be equal to one."
        self._probs = p.copy()
        self._n_class = p.size
        self._setup_alias_method()

    def _setup_alias_method(self):
        self._box_index = np.full(shape=self._n_class, fill_value=-1, dtype=np.int32)
        self._box_threshold = np.full(shape=self._n_class, fill_value=np.nan, dtype=np.float64)
        self._box_height = box_height = np.mean(self._probs)

        _probs = self._probs.copy()
        top = np.where(_probs > box_height)[0]
        bottom = np.where(_probs <= box_height)[0]
        for _ in range(self._n_class):

            idx_bottom = bottom[-1]
            if top.size == 0:
                self._box_index[idx_bottom] = idx_bottom
                self._box_threshold[idx_bottom] = box_height
            else:
                idx_top = top[-1]
                self._box_index[idx_bottom] = idx_top
                self._box_threshold[idx_bottom] = _probs[idx_bottom]
                _probs[idx_top] -= (box_height - _probs[idx_bottom])

            # update
            bottom = np.delete(bottom, -1)

            if (_probs[idx_top] <= box_height or bottom.size == 0) and top.size > 0:
                bottom = np.append(bottom, top[-1])
                top = np.delete(top, -1)

    def random(self, size: int, replace: bool = True):

        assert replace, "currently it doesn't support sampling without replacement(replace=`False`)"

        # sample from box
        sampled = np.random.randint(self._n_class, size=size)
        # flip if it exceeds threshold
        rand_flip = np.random.rand(size) * self._box_height
        flip_or_not = rand_flip > self._box_threshold[sampled]
        # flip samples that exceed threshold
        sampled[flip_or_not] = self._box_index[sampled][flip_or_not]

        return sampled


