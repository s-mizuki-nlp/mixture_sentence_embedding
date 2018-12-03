#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import norm

def generate_random_orthogonal_vectors(n_dim: int, n_vector: int, l2_norm: float):
    """
    generate random orthogonal vectors.
    l2 norm of each vector will be `l2_norm`
    distance between each vector pair will be `\sqr{2}*l2_norm`

    :param n_dim: vector length
    :param n_vector: number of vectors
    :param l2_norm: l2(ret[i])
    :return: generated random vectors
    """
    x = np.random.normal(size=n_dim * n_vector).reshape((n_dim, n_vector))
    mat_u, _, _ = np.linalg.svd(x, full_matrices=False)

    mat_ret = mat_u.T
    mat_ret = mat_ret / np.linalg.norm(mat_ret, axis=1, keepdims=True) * l2_norm

    return mat_ret


def calculate_prior_dist_params(expected_swd: float, n_dim_latent: int):
    c = 1.22
    l2_norm = np.sqrt(0.5*c*np.sqrt(n_dim_latent)*expected_swd)
    std = l2_norm / np.sqrt(n_dim_latent) + 1.

    return l2_norm, std