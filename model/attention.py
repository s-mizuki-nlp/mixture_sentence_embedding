#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class SimpleGlobalAttentionUsingRNNHiddenState(nn.Module):

    def __init__(self, dim_query: int, dim_key: int):

        super(__class__, self).__init__()

        self._transform_query = nn.Linear(in_features=dim_query, out_features=dim_key, bias=False)

        self._dim_query = dim_query
        self._dim_key = dim_key

    def forward(self, source: Tensor, memory_bank: Tensor) -> (Tensor, Tensor):
        """

        :param source: output of RNN(=lstm, gru) state: (N_layer x Is_bidirectional, N_batch, N_dim_lstm)
        :param memory_bank: set of context vector: (N_batch, N_context, N_dim_key)
        :return:
        """
        # reshape query
        # v_q = (N_batch, N_dim_rnn x N_layer x Is_bidirectional = N_dim_query)  n_mb = source.shape[1]
        n_mb = source.shape[1]
        v_q = source.permute([1,0,2]).contiguous().view(n_mb, -1)

        # transform
        # v_q = (N_batch, N_dim_key)
        v_q = self._transform_query.forward(v_q)
        # unsqueeze
        # v_q = (N_batch, 1, N_dim_key)
        v_q = v_q.unsqueeze(dim=1)

        # calculate attention
        # h_t = memory_bank = (N_batch, N_context, N_dim_key)
        # h_s = (N_batch, N_dim_key, 1)
        h_s = v_q.permute([0,2,1]).contiguous()
        h_t = memory_bank
        v_score = torch.bmm(h_t, h_s)

        # v_score, v_attn = (N_batch, N_context, 1)
        v_attn = F.softmax(v_score, dim=-1)

        # calculate context vector
        # v_context = (N_batch, N_dim_key), v_attn_score = (N_batch, N_context)
        v_context = torch.sum(h_t * v_attn, dim=1, keepdim=False)
        v_attn = v_attn.squeeze(dim=-1)

        return v_context, v_attn

class GlobalAttentionUsingRNNHiddenState(nn.Module):

    def __init__(self, dim_query: int, dim_key: int, attn_type: str, attn_func: str = "softmax"):

        from onmt.modules import GlobalAttention

        assert attn_type in ["general", "mlp"], f"invalid attention type: {attn_type}"

        super(__class__, self).__init__()

        self._transform_query = nn.Linear(in_features=dim_query, out_features=dim_key, bias=False)
        _attn_type = "dot" if attn_type == "general" else attn_type
        self._global_attention = GlobalAttention(dim=dim_key, attn_type=_attn_type, attn_func=attn_func)

        self._dim_query = dim_query
        self._dim_key = dim_key
        self._attn_type = attn_type
        self._attn_func = attn_func

    def forward(self, source: Tensor, memory_bank: Tensor, memory_lengths=None, coverage=None) -> (Tensor, Tensor):
        """

        :param source: output of RNN(=lstm, gru) state: (N_layer x Is_bidirectional, N_batch, N_dim_lstm)
        :param memory_bank: set of context vector: (N_batch, N_context, N_dim_key)
        :param memory_lengths: optional
        :param coverage: optional

        :return:
        """
        # reshape query
        # v_q = (N_batch, N_dim_rnn x N_layer x Is_bidirectional = N_dim_query)
        n_mb = source.shape[1]
        v_q = source.permute([1,0,2]).contiguous().view(n_mb, -1)

        # transform
        # v_q = (N_batch, N_dim_key
        v_q = self._transform_query.forward(v_q)
        # unsqueeze
        # v_q = (N_batch, 1, N_dim_key)
        v_q = v_q.unsqueeze(dim=1)

        # calculate attention
        # v_context = (N_batch, N_dim_key), v_attn_score = (N_batch, N_context)
        v_context, v_attn_score = self._global_attention.forward(source=v_q, memory_bank=memory_bank, memory_lengths=memory_lengths, coverage=coverage)
        v_context = v_context.squeeze(dim=0)
        v_attn_score = v_attn_score.squeeze(dim=0)

        return v_context, v_attn_score