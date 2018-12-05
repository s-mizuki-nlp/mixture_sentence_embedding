#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional

import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
import numpy as np


class GMMSampler(nn.Module):

    def __init__(self, n_sample: int, param_tau: float, expect_log_alpha: bool = False, device=torch.device("cpu"), debug=False):

        super().__init__()

        self._n_sample = n_sample
        self._tau = param_tau
        self._debug = debug
        self._device = device
        self._expect_log_alpha = expect_log_alpha

    @property
    def tau(self) -> float:
        return self._tau

    @tau.setter
    def tau(self, value: float):
        self._tau = value

    @property
    def sample_size(self):
        return self._n_sample

    @sample_size.setter
    def sample_size(self, value: int):
        self._n_sample = value

    @property
    def expect_log_alpha(self):
        return self._expect_log_alpha

    def _sample_gumbel(self, n_dim: int, size: Optional[int] = None):
        """
        returns random number(size, n_dim) sampled from gumbel distribution

        :param n_dim: number of dimensions
        :param size: number of samples
        """
        size = self._n_sample if size is None else size

        rand_unif = np.random.uniform(size=(size, n_dim)).astype(np.float32)
        v = np.log(-np.log(rand_unif))
        t = torch.tensor(data=v, dtype=torch.float, requires_grad=False, device=self._device)

        return t

    def _sample_normal(self, n_dim: int, size: Optional[int] = None):
        """
        returns random number(size, n_dim) sampled from standard normal distribution

        :param n_dim: number of dimensions
        :param size: number of samples
        """
        size = self._n_sample if size is None else size

        v = np.random.normal(size=(size, n_dim)).astype(np.float32)
        t = torch.tensor(data=v, dtype=torch.float, requires_grad=False, device=self._device)

        return t

    def _reparametrization_trick(self, vec_alpha: torch.Tensor, mat_mu: torch.Tensor, mat_std: torch.Tensor):
        n_len, n_dim = mat_mu.size()

        # v_epsilon = (n_len, n_dim)
        # v_epsilon[t,:] ~ Normal(0,1)
        v_epsilon = self._sample_normal(size=n_len, n_dim=n_dim)
        # mat_z = (n_len, n_dim)
        # mat_z[t] = alpha[t] * (\mu[t] + \std[t] * \epsilon[t])
        mat_z = (mat_mu + mat_std * v_epsilon) * vec_alpha.unsqueeze(-1)
        # vec_z = \sum_{t}{ alpha[t] * (\mu[t] + \std[t] * \epsilon[t]) }
        vec_z = mat_z.sum(dim=0)

        return vec_z

    def forward(self, lst_vec_alpha_component: List[torch.Tensor], lst_mat_mu: List[torch.Tensor], lst_mat_std: List[torch.Tensor]):
        """
        sample `n_sample` random samples from gaussian mixture using both gumbel-softmax trick and re-parametrization trick.

        :param lst_vec_alpha_component: list of the packed scale vectors. if self.expect_log_alpha=True, input must be log-scale vectors.
        :param lst_mat_mu: list of the sequence of packed mean vectors
        :param lst_mat_std: list of the sequence of packed standard deviation vectors(=sqrt of the diagonal elements of covariance matrix)
        :return: random samples with shape (n_mb, n_sample, n_dim)
        """

        if self._expect_log_alpha:
            lst_vec_ln_alpha = lst_vec_alpha_component
        else:
            lst_vec_ln_alpha = [torch.log(vec_alpha) for vec_alpha in lst_vec_alpha_component]
        # gumbel-softmax trick + re-parametrization trick
        lst_mat_z = []
        lst_mat_alpha = []
        # vec_ln_alpha = (n_len,), mat_mu = (n_len, n_dim), mat_std = (n_len, n_dim)
        for vec_ln_alpha, mat_mu, mat_std in zip(lst_vec_ln_alpha, lst_mat_mu, lst_mat_std):
            # apply gumbel-softmax on vec_ln_alpha
            # mat_alpha = (n_sample, n_len)
            mat_alpha = gumbel_softmax(logits=vec_ln_alpha.repeat((self._n_sample,1)), tau=self._tau)
            if self._debug:
                lst_mat_alpha.append(mat_alpha)

            # apply re-parametrization trick
            iter_rp = [self._reparametrization_trick(vec_alpha, mat_mu, mat_std) for vec_alpha in mat_alpha]
            # mat_z = (n_sample, n_dim)
            mat_z = torch.stack(iter_rp)
            lst_mat_z.append(mat_z)

        # tensor_z = (n_mb, n_sample, n_dim)
        tensor_z = torch.stack(lst_mat_z)

        if self._debug:
            return lst_mat_alpha, lst_mat_z
        else:
            return tensor_z