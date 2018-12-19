#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Tuple, Union, Iterator, Optional
from .mixture import MultiVariateGaussianMixture

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


def _kldiv_diag(mu_x: np.array, cov_x: np.ndarray, mu_y: np.array, cov_y: np.ndarray):
    """
    calculate KL(f_x||f_y); kullback-leibler divergence between two gaussian distributions parametrized by $mu,\Sigma$
    but $\Sigma$ is diagonal matrix.

    :param mu_x: mean vector of x
    :param cov_x: covariance matrix of x
    :param mu_y: mean vector of y
    :param cov_y: covariance matrix of y
    """
    n_dim = mu_x.size
    vec_nu_x = np.diag(cov_x)
    vec_nu_y = np.diag(cov_y)

    det_term = np.sum(np.log(vec_nu_y)) - np.sum(np.log(vec_nu_x))
    tr_term = np.sum(vec_nu_x / vec_nu_y)
    quad_term = np.sum( (mu_x - mu_y)**2 / vec_nu_y )

    kldiv = 0.5*(det_term + tr_term - n_dim + quad_term)

    return kldiv


def approx_kldiv_between_diag_gmm(p_x: MultiVariateGaussianMixture, p_y: MultiVariateGaussianMixture) -> float:
    """
    calculates approximated KL(p_x||p_y); kullback-leibler divergence between two gaussian mixtures parametrized by $\{\alpha_k, \mu_k,\Sigma_k\}$.
    but all $\Sigma_k$ is diagonal matrix.

    :param p_x: instance of MultiVariateGaussianMixture class.
    :param p_y: instance of MultiVariateGaussianMixture class.
    """
    assert p_x.is_cov_diag and p_y.is_cov_diag, "both GMM must have diagonal covariance matrix."

    n_c_x, n_c_y = p_x.n_component, p_y.n_component
    vec_ln_term = np.zeros(n_c_x, dtype=np.float64)
    for c_x in range(n_c_x): # M
        alpha_c_x, mu_c_x, cov_c_x = p_x._alpha[c_x], p_x._mu[c_x], p_x._cov[c_x] # j=1,2,...,M
        sum_pi_exp_c_x = np.sum([p_x._alpha[c]*np.exp(-_kldiv_diag(mu_c_x, cov_c_x, p_x._mu[c], p_x._cov[c])) for c in range(n_c_x)])
        sum_pi_exp_c_x_y = np.sum([p_y._alpha[c]*np.exp(-_kldiv_diag(mu_c_x, cov_c_x, p_y._mu[c], p_y._cov[c])) for c in range(n_c_y)])
        vec_ln_term[c_x] = np.log(sum_pi_exp_c_x) - np.log(sum_pi_exp_c_x_y)

    kldiv = np.sum(p_x._alpha * vec_ln_term)

    return kldiv


def mc_kldiv_between_diag_gmm(p_x: MultiVariateGaussianMixture, p_y: MultiVariateGaussianMixture, n_sample=int(1E5)) -> float:
    """
    calculates approximated KL(p_x||p_y); kullback-leibler divergence between two gaussian mixtures parametrized by $\{\alpha_k, \mu_k,\Sigma_k\}$.
    but all $\Sigma_k$ is diagonal matrix.

    :param p_x: instance of MultiVariateGaussianMixture class.
    :param p_y: instance of MultiVariateGaussianMixture class.
    """
    vec_x = p_x.random(size=n_sample)
    kldiv = np.mean(p_x.logpdf(vec_x) - p_y.logpdf(vec_x))

    return kldiv
