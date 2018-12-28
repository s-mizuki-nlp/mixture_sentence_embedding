#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class SimpleGlobalAttention(nn.Module):

    def __init__(self, n_dim_query: int, n_dim_memory: int):

        super(__class__, self).__init__()

        self._transform_query = nn.Linear(in_features=n_dim_query, out_features=n_dim_memory, bias=False)

        self._dim_query = n_dim_query
        self._dim_key = n_dim_memory

    def forward(self, source: Tensor, memory_bank: Tensor) -> (Tensor, Tensor):
        """

        :param source: 2-D tensor (N_batch, N_dim_query). typically, output of RNN(=lstm, gru) state: (N_batch, N_dim_state)
        :param memory_bank: set of context vector: (N_batch, N_context, N_dim_key)
        :return:
        """
        # v_q = (N_batch, N_dim_query)
        v_q = source

        # transform
        # v_q = (N_batch, N_dim_key)
        v_q = self._transform_query.forward(v_q)
        # unsqueeze
        # v_q = (N_batch, 1, N_dim_key)
        v_q = v_q.unsqueeze(dim=1)

        # calculate attention
        # h_t = memory_bank = (N_batch, N_context, N_dim_key)
        # h_s = (N_batch, N_dim_key, 1)
        h_t = memory_bank
        h_s = v_q.permute([0,2,1]).contiguous()
        # v_score = (N_batch, N_context, 1)
        v_score = torch.bmm(h_t, h_s)

        # v_attn = (N_batch, N_context, 1)
        v_attn = F.softmax(v_score, dim=1)

        # calculate context vector
        # v_context = (N_batch, N_dim_key), v_attn = (N_batch, N_context)
        v_context = torch.sum(h_t * v_attn, dim=1, keepdim=False)
        v_attn = v_attn.squeeze(dim=-1)

        return v_context, v_attn
