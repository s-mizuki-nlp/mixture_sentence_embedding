#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .attention import SimpleGlobalAttention

class SelfAttentiveLSTMDecoder(nn.Module):

    def __init__(self, n_dim_lstm_hidden: int, n_dim_memory: int,
                 attention_layer: Optional[SimpleGlobalAttention] = None):

        super(__class__, self).__init__()

        self._n_dim_lstm_hidden = n_dim_lstm_hidden
        self._n_dim_memory = n_dim_memory

        if attention_layer is not None:
            self._attention_layer = attention_layer
        else:
            n_dim_query = n_dim_lstm_hidden * 2
            self._attention_layer = SimpleGlobalAttention(dim_query=n_dim_query, dim_key=n_dim_memory)

        self._lstm_cell = nn.LSTMCell(input_size=n_dim_memory, hidden_size=n_dim_lstm_hidden)

    def _init_state(self, size: int):
        h_0 = torch.zeros(size, self._n_dim_lstm_hidden)
        c_0 = torch.zeros(size, self._n_dim_lstm_hidden)
        return h_0, c_0

    def forward(self, z_latent: torch.Tensor, n_step: int) -> torch.Tensor:
        """
        completely input-less sequence decoder.

        :param z_latent: latent tensor; set of latent vectors. shape = (N_batch, N_sample, N_dim_latent)
        :param n_step: maximum sequence length

        :return tensor of output state. shape = (N_batch, N_step, N_dim_lstm_hidden)
        """
        n_batch = z_latent.shape[0]
        h_t, c_t = self._init_state(size=n_batch)
        lst_h_t = []

        # calculate each step
        for t in range(n_step):
            # state vector
            v_t = torch.cat((h_t,c_t), dim=-1)
            # calculate context vector using attention layer with state vector as query
            x_dec_t, x_dec_t_attn = self._attention_layer.forward(source=v_t, memory_bank=z_latent)
            # calculate next step
            h_t, c_t = self._lstm_cell(x_dec_t, (h_t, c_t))
            # store output state vector
            lst_h_t.append(h_t)

        # stack it
        t_dec_h = torch.stack(lst_h_t, dim=1)

        return t_dec_h


class SimplePrediction(nn.Module):

    def __init__(self, n_dim_in: int, n_dim_out: int, log: bool = False, bias: bool = True,
                 shared_weight: Optional[nn.Parameter] = None):

        super(__class__, self).__init__()

        self._linear = nn.Linear(in_features=n_dim_in, out_features=n_dim_out, bias=bias)
        if shared_weight is not None:
            self._linear.weight = shared_weight

        if log:
            self._activation = F.log_softmax
        else:
            self._activation = F.softmax

    def forward(self, x_input):

        x_to_y = self._linear.forward(x_input)
        y = self._activation(x_to_y, dim=-1)

        return y