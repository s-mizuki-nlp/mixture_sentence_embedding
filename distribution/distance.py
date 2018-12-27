#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.misc import logsumexp
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Tuple, Union, Iterator, Optional
from .mixture import MultiVariateGaussianMixture


# Earth Mover's Distance.
def earth_mover_distance(vec_p: np.array, vec_q: np.array, mat_dist: Optional[np.ndarray] = None,
                         mat_x: Optional[np.ndarray] = None, mat_y: Optional[np.ndarray] = None,
                         lambda_: float = 0.1, epsilon: float = 0.01, n_iter_max: int = 20,
                         return_optimal_transport: bool = False):
    """
    calculate earth mover's distance between two point-mass distribution (ex. set of word vectors)
    """
    if mat_dist is None:
        assert (mat_x is not None) and (mat_y is not None), "you must specify either `mat_dist` or `(mat_x, mat_y)` pair."
        mat_dist = euclidean_distances(mat_x, mat_y)
    assert vec_p.size == mat_dist.shape[0], "mat_dist.shape must be identical to (vec_p.size, vec_q.size)"
    assert vec_q.size == mat_dist.shape[1], "mat_dist.shape must be identical to (vec_p.size, vec_q.size)"

    vec_ln_p = np.log(vec_p)
    vec_ln_q = np.log(vec_q)
    vec_ln_a = np.zeros_like(vec_p, dtype=np.float) # vec_ln_p.copy()
    vec_ln_b = np.zeros_like(vec_q, dtype=np.float) # vec_ln_q.copy()
    mat_ln_k = -mat_dist / lambda_

    for n_iter in range(n_iter_max):

        vec_ln_a = vec_ln_p - logsumexp(mat_ln_k + vec_ln_b.reshape(1,-1), axis=1)
        vec_ln_b = vec_ln_q - logsumexp(mat_ln_k.T + vec_ln_a.reshape(1,-1), axis=1)

        # termination
        ## difference with condition
        mat_gamma = np.exp(vec_ln_a.reshape(-1,1) + mat_ln_k + vec_ln_b.reshape(1,-1))
        err = np.mean(np.abs(mat_gamma.sum(axis=1) - vec_p))
        if err < epsilon:
            break

    dist = np.sum(mat_gamma*mat_dist)

    if return_optimal_transport:
        return dist, mat_gamma
    else:
        return dist


def wasserstein_distance_sq_between_gmm(p_x: MultiVariateGaussianMixture, p_y: MultiVariateGaussianMixture, return_distance_matrix=False, **kwargs):
    """
    wasserstein distance between gussian mixtures.
    """
    assert p_x.n_dim == p_y.n_dim, "dimension size mismatch detected."
    assert p_x.is_cov_diag and p_y.is_cov_diag, "it supports gmm with diagonal covariance only."

    vec_p, vec_q = p_x._alpha, p_y._alpha
    n_c_x, n_c_y = p_x.n_component, p_y.n_component
    mat_dist = np.zeros(shape=(n_c_x, n_c_y), dtype=np.float)
    for i in range(n_c_x):
        for j in range(n_c_y):
            mat_dist[i,j] = _wasserstein_distance_sq_between_multivariate_normal_diag(
                vec_mu_x=p_x._mu[i], vec_std_x=np.sqrt(p_x._cov[i]), vec_mu_y=p_y._mu[j], vec_std_y=np.sqrt(p_y._cov[j])
            )
    # in case you don't need minimization
    if n_c_x == 1: # vec_p = [1]
        return mat_dist.dot(vec_q)
    if n_c_y == 1: # vec_q = [1]
        return vec_p.dot(mat_dist)

    wd = earth_mover_distance(vec_p=vec_p, vec_q=vec_q, mat_dist=mat_dist, **kwargs)

    if return_distance_matrix:
        return wd, mat_dist
    else:
        return wd


def _wasserstein_distance_sq_between_multivariate_normal(vec_mu_x: np.array, mat_cov_x: np.ndarray, vec_mu_y: np.array, mat_cov_y: np.ndarray) -> float:
    """
    wasserstein distance between multivariate normal distributions
    """
    d_mu = np.sum((vec_mu_x - vec_mu_y) ** 2)
    mat_std_x = sqrtm(mat_cov_x)
    d_cov = np.sum(np.diag(mat_cov_x + mat_cov_y - 2*np.sqrt(mat_std_x*mat_cov_y*mat_std_x)))

    return d_mu + d_cov

# wasserstein distance between multivariate normal with diagonal covariance matrix
def _wasserstein_distance_sq_between_multivariate_normal_diag(vec_mu_x: np.array, vec_std_x: np.array, vec_mu_y: np.array, vec_std_y: np.array) -> float:
    """
    wasserstein distance between multivariate normal distributions with diagonal covariance matrix
    """
    d_mu = np.sum((vec_mu_x - vec_mu_y)**2)
    d_cov = np.sum((vec_std_x - vec_std_y)**2)

    return d_mu + d_cov

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
        # 2018-12-27 re-implemented using logsumexp() function
        log_sum_pi_exp_c_x = logsumexp(np.log(p_x._alpha) - np.array([_kldiv_diag(mu_c_x, cov_c_x, p_x._mu[c], p_x._cov[c]) for c in range(n_c_x)]))
        log_sum_pi_exp_c_x_y = logsumexp(np.log(p_y._alpha) - np.array([_kldiv_diag(mu_c_x, cov_c_x, p_y._mu[c], p_y._cov[c]) for c in range(n_c_y)]))
        vec_ln_term[c_x] = log_sum_pi_exp_c_x - log_sum_pi_exp_c_x_y
        # sum_pi_exp_c_x = np.sum([p_x._alpha[c]*np.exp(-_kldiv_diag(mu_c_x, cov_c_x, p_x._mu[c], p_x._cov[c])) for c in range(n_c_x)])
        # sum_pi_exp_c_x_y = np.sum([p_y._alpha[c]*np.exp(-_kldiv_diag(mu_c_x, cov_c_x, p_y._mu[c], p_y._cov[c])) for c in range(n_c_y)])
        # vec_ln_term[c_x] = np.log(sum_pi_exp_c_x) - np.log(sum_pi_exp_c_x_y)

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
    vec_x_ln_p = p_x.logpdf(vec_x)
    vec_y_ln_p = p_y.logpdf(vec_x)
    vec_w = np.exp(vec_x_ln_p - logsumexp(vec_x_ln_p))
    kldiv = np.sum( vec_w * (vec_x_ln_p - vec_y_ln_p) )

    return kldiv
