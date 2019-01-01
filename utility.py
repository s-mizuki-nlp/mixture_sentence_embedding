#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Union, Optional, Dict
import torch
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from distribution.mixture import MultiVariateGaussianMixture
from distribution.distance import approx_kldiv_between_diag_gmm, mc_kldiv_between_diag_gmm

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


def calculate_prior_dist_params(expected_wd: float, n_dim_latent: int, sliced_wasserstein: bool):
    if sliced_wasserstein:
        c = 1.22
        l2_norm = np.sqrt(0.5 * c * np.sqrt(n_dim_latent) * expected_wd)
    else:
        l2_norm = np.sqrt(0.5 * expected_wd)

    std = l2_norm / np.sqrt(n_dim_latent) + 1.
    return l2_norm, std


def calculate_mean_l2_between_sample(t_z_posterior: np.ndarray):
    
    n_mb, n_latent, n_dim = t_z_posterior.shape
    if n_latent == 1:
        return 0.

    idx_triu = np.triu_indices(n=n_latent, k=1)
    l2_dist = 0.
    for b in range(n_mb):
        mat_z_b = t_z_posterior[b]
        mat_dist_b = euclidean_distances(mat_z_b, mat_z_b)
        l2_dist = l2_dist + np.mean(mat_dist_b[idx_triu])

    l2_dist /= n_mb

    return l2_dist


def _tensor_to_array(t: Union[np.ndarray, torch.Tensor]):
    if isinstance(t, torch.Tensor):
        return t.cpu().data.numpy()
    elif isinstance(t, np.ndarray):
        return t
    else:
        raise AttributeError(f"unsupported type: {type(t)}")

def calculate_kldiv(lst_v_alpha: List[Union[np.ndarray, torch.Tensor]], lst_v_mu: List[Union[np.ndarray, torch.Tensor]],
                    lst_v_sigma: List[Union[np.ndarray, torch.Tensor]],
                    prior_distribution: MultiVariateGaussianMixture,
                    method: str = "analytical", n_mc_sample: Optional[int] = None):
    available_method = "analytical,monte_carlo"
    assert method in available_method.split(","), f"`method` must be one of these: {available_method}"

    n_mb = len(lst_v_alpha)
    iter_alpha = map(_tensor_to_array, lst_v_alpha)
    iter_mu = map(_tensor_to_array, lst_v_mu)
    iter_sigma = map(_tensor_to_array, lst_v_sigma)
    kldiv = 0.0
    for alpha, mu, sigma in zip(iter_alpha, iter_mu, iter_sigma):
        n_dim_sigma = sigma.shape[-1]
        if n_dim_sigma == 1: # istropic covariance matrix
            posterior = MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma)
        else: # diagonal covariance matrix
            posterior = MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, mat_cov=sigma**2)

        if method == "analytical":
            kldiv_b = approx_kldiv_between_diag_gmm(p_x=posterior, p_y=prior_distribution)
        else:
            kldiv_b = mc_kldiv_between_diag_gmm(p_x=posterior, p_y=prior_distribution, n_sample=n_mc_sample)
        kldiv += kldiv_b

    kldiv /= n_mb

    return kldiv
