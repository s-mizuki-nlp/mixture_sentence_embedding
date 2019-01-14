#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io, os
from typing import List, Union, Optional, Dict
import warnings
import copy
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

    if n_vector == 1:
        warnings.warn("single vector was requested. it will return zero vector.")
        mat_ret = np.zeros((1, n_dim), dtype=np.float64)
    else:
        if l2_norm == 0:
            warnings.warn("vectors with zero length was requested. it will return zero vectors.")
            mat_ret = np.zeros((n_vector, n_dim), dtype=np.float64)
        else:
            if n_vector <= n_dim:
                warnings.warn("multiple vectors were requested. it will return orthogonal vector set with specified norm.")
                np.random.seed(seed=0)
                x = np.random.normal(size=n_dim * n_vector).reshape((n_dim, n_vector))
                mat_u, _, _ = np.linalg.svd(x, full_matrices=False)
                mat_ret = mat_u.T
            else:
                warnings.warn("multiple vectors were requested. it will return random vector set with specified norm.")
                mat_ret = np.random.normal(size=n_dim * n_vector).reshape((n_vector, n_dim))

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
                    method: str = "analytical", n_mc_sample: Optional[int] = None,
                    return_list: bool = False):
    available_method = "analytical,monte_carlo"
    assert method in available_method.split(","), f"`method` must be one of these: {available_method}"

    n_mb = len(lst_v_alpha)
    iter_alpha = map(_tensor_to_array, lst_v_alpha)
    iter_mu = map(_tensor_to_array, lst_v_mu)
    iter_sigma = map(_tensor_to_array, lst_v_sigma)
    kldiv = []
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
        kldiv.append(kldiv_b)

    if return_list:
        return kldiv
    else:
        ret = np.mean(kldiv)
        return ret


def sigmoid_generator(scale: float, coef: float, offset: float, min_value=1E-4):
    """
    it returns scale*sigmoid(coef*(x-offset)) = scale / (1. + exp(-coef*(x-offset)))
    returned value range will be [min_value, scale]
    """

    _s = scale
    _c = coef
    _o = offset
    _m = min_value

    def sigmoid(x: Union[int, float]):
        ret = _s / (1. + np.exp(-_c*(x - _o)))
        ret = min(max(_m, ret), _s)
        return ret

    return sigmoid


def enumerate_optional_metrics(cfg_metrics: Union[List[str], Dict[str,int]], n_epoch):

    if isinstance(cfg_metrics, list):
        return copy.deepcopy(cfg_metrics)
    elif isinstance(cfg_metrics, dict):
        lst_ret = []
        for metric, n_interval in cfg_metrics.items():
            if n_epoch % n_interval == 0:
                lst_ret.append(metric)
        return lst_ret
    else:
        raise NotImplementedError("unsupported metrics configuration detected:", cfg_metrics)


def write_log_and_progress(n_epoch, n_processed, mode: str, dict_metrics, logger: io.TextIOWrapper,
                         output_log: bool, output_std: bool):

    func_value_to_str = lambda v: f"{v:1.7f}" if isinstance(v,float) else f"{v}"

    metrics = {
        "epoch":n_epoch,
        "processed":n_processed,
        "mode":mode
    }
    metrics.update(dict_metrics)

    if output_log:
        sep = "\t"
        ## output log file
        if os.stat(logger.name).st_size == 0:
            s_header = sep.join(metrics.keys()) + "\n"
            logger.write(s_header)
        else:
            s_record = sep.join( map(func_value_to_str, metrics.values()) ) + "\n"
            logger.write(s_record)
        logger.flush()

    ## output metrics
    if output_std:
        prefix = metrics["mode"]
        s_print = ", ".join( [f"{prefix}_{k}:{func_value_to_str(v)}" for k,v in metrics.items()] )
        print(s_print)
