#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Tuple, Union, Iterator, Optional

def earth_mover_distance(vec_p: np.array, vec_q: np.array, mat_dist: Optional[np.ndarray] = None,
                         mat_x: Optional[np.ndarray] = None, mat_y: Optional[np.ndarray] = None,
                         lambda_: float = 0.1, mean_err_threshold: float = 1E-5, return_transport: bool = False):
    if mat_dist is None:
        assert (mat_x is not None) and (mat_y is not None), "you must specify either `mat_dist` or `(mat_x, mat_y)` pair."
        mat_dist = euclidean_distances(mat_x, mat_y)
    assert vec_p.size == mat_dist.shape[0], "mat_dist.shape must be identical to (vec_p.size, vec_q.size)"
    assert vec_q.size == mat_dist.shape[1], "mat_dist.shape must be identical to (vec_p.size, vec_q.size)"

    vec_a = np.ones_like(vec_p)
    vec_b = np.ones_like(vec_q)
    mat_k = np.exp(-mat_dist/lambda_)

    while True:
        vec_a = vec_p / mat_k.dot(vec_b)
        vec_b = vec_q / mat_k.T.dot(vec_a)

        mat_gamma = vec_a.reshape((-1,1)) * mat_k * vec_b.reshape((1,-1))
        err = np.mean(np.abs(np.sum(mat_gamma, axis=1) - vec_p))
        if err < mean_err_threshold:
            break

    dist = np.sum(mat_gamma*mat_dist)

    if return_transport:
        return dist, mat_gamma
    else:
        return dist