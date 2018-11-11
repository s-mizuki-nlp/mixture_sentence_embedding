#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from typing import Optional
from model.encoder import GMMLSTMEncoder
from model.noise_layer import GMMSampler
from model.decoder import SelfAttentiveLSTMDecoder
from preprocess import utils

class VariationalAutoEncoder(nn.Module):

    def __init__(self,
                   seq_to_gmm_encoder: GMMLSTMEncoder,
                   gmm_sampler: GMMSampler,
                   set_to_state_decoder: SelfAttentiveLSTMDecoder,
                   state_to_seq_decoder: nn.Module
                   ):

        super(__class__, self).__init__()
        self._encoder = seq_to_gmm_encoder
        self._sampler = gmm_sampler
        self._decoder = set_to_state_decoder
        self._predictor = state_to_seq_decoder

        assert seq_to_gmm_encoder._apply_softmax, "it expects scaled \alpha output."

    @property
    def sampler_tau(self):
        return self._sampler.tau

    @sampler_tau.setter
    def sampler_tau(self, value: float):
        self._sampler.tau = value

    @property
    def sampler_size(self):
        return self._sampler.sample_size

    def forward(self, x_seq: torch.Tensor, x_seq_len: torch.Tensor):
        """

        :param x_seq:
        :param x_seq_len:
        :return:  v_alpha, v_mu, v_sigma, v_z, v_ln_prob_y
        """

        # encoder: Sequence to padded GMM parameters {\alpha, \mu, \sigma}
        v_alpha, v_mu, v_sigma, (h_n, c_n) = self._encoder.forward(x_seq, x_seq_len)

        # pack padded sequence while keeping torch.tensor object
        lst_seq_len = x_seq_len.data.numpy()
        lst_alpha = utils.pack_padded_sequence(v_alpha, lst_seq_len=lst_seq_len, dim=0, keep_torch_tensor=True)
        lst_mu = utils.pack_padded_sequence(v_mu, lst_seq_len=lst_seq_len, dim=0, keep_torch_tensor=True)
        lst_sigma = utils.pack_padded_sequence(v_sigma, lst_seq_len=lst_seq_len, dim=0, keep_torch_tensor=True)

        # sample from posterior distribution
        v_z = self._sampler.forward(lst_vec_alpha=lst_alpha, lst_mat_mu=lst_mu, lst_mat_std=lst_sigma)

        # decode from latent representation vector set
        n_step = max(lst_seq_len)
        v_dec_h = self._decoder.forward(z_latent=v_z, n_step=n_step)

        # predict log probability of output tokens
        v_ln_prob_y = self._predictor.forward(v_dec_h)

        # return everything
        return v_alpha, v_mu, v_sigma, v_z, v_ln_prob_y