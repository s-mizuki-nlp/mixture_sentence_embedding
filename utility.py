#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import norm

def generate_random_orthogonal_vectors(N_dim: int, N_vector: int, dist: float):
    """
    generate random orthogonal vectors.
    l2 distance between each vector will be `dist`

    :param N_dim: vector length
    :param N_vector: number of vectors
    :param dist: l2(ret[i], ret[j])
    :return: generated random vectors
    """
    x = np.random.normal(size=N_dim*N_vector).reshape((N_dim, N_vector))
    mat_u, _, _ = np.linalg.svd(x, full_matrices=False)

    mat_ret = mat_u.T
    mat_ret = mat_ret / np.linalg.norm(mat_ret, axis=1, keepdims=True) * np.sqrt(0.5*dist)

    return mat_ret