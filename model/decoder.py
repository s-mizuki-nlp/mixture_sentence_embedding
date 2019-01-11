#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .attention import SimpleGlobalAttention

class SelfAttentiveLSTMDecoder(nn.Module):

    def __init__(self, n_dim_lstm_hidden: int, n_dim_lstm_input: int,
                 latent_decoder: SimpleGlobalAttention,
                 n_lstm_layer: int = 1,
                 device=torch.device("cpu")):

        super(__class__, self).__init__()

        self._n_dim_lstm_hidden = n_dim_lstm_hidden
        self._n_dim_lstm_input = n_dim_lstm_input
        self._n_lstm_layer = n_lstm_layer
        self._attention_layer = latent_decoder
        self._device = device

        # instanciate lstm cell
        self._lst_lstm_cell = []
        for k in range(n_lstm_layer):
            if k == 0:
                self._lst_lstm_cell.append(nn.LSTMCell(input_size=n_dim_lstm_input, hidden_size=n_dim_lstm_hidden))
            else:
                self._lst_lstm_cell.append(nn.LSTMCell(input_size=n_dim_lstm_hidden, hidden_size=n_dim_lstm_hidden))
        self._lstm_layers = nn.ModuleList(self._lst_lstm_cell)


    def _init_state(self, size: int):
        h_0 = torch.zeros(size, self._n_dim_lstm_hidden, device=self._device)
        c_0 = torch.zeros(size, self._n_dim_lstm_hidden, device=self._device)
        return h_0, c_0

    @property
    def n_dim_memory(self):
        return self._n_dim_lstm_input

    def forward(self, z_latent: torch.Tensor, n_step: int) -> torch.Tensor:
        """
        completely input-less sequence decoder.

        :param z_latent: latent tensor; set of latent vectors. shape = (N_batch, N_sample, N_dim_latent)
        :param n_step: maximum sequence length

        :return tensor of output state. shape = (N_batch, N_step, N_dim_lstm_hidden)
        """
        n_batch = z_latent.shape[0]
        lst_h_t = [None]*n_step

        # calculate each layers
        for k, lstm_cell_k in enumerate(self._lstm_layers):
            # initialie internal state tensors
            h_t, c_t = self._init_state(size=n_batch)

            # calculate each steps
            for t in range(n_step):

                if k == 0:
                    # re-use internal states as the query
                    q_t = torch.cat((h_t,c_t), dim=-1)
                    # calculate context vector using attention layer with internal states as query
                    x_dec_t, x_dec_t_attn = self._attention_layer.forward(source=q_t, memory_bank=z_latent)
                else:
                    # just retrieve output of the previous layer
                    x_dec_t = lst_h_t[t]
                # calculate next step
                h_t, c_t = lstm_cell_k(x_dec_t, (h_t, c_t))

                # store output state vector: h_t
                lst_h_t[t] = h_t

        # stack it
        t_dec_h = torch.stack(lst_h_t, dim=1)

        return t_dec_h


class SimplePredictor(nn.Module):

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