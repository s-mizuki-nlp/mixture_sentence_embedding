#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from typing import Optional
import warnings

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

class SimpleGlobalAttention(nn.Module):

    def __init__(self, n_dim_query: int, n_dim_memory: int):

        super(__class__, self).__init__()

        self._transform_query = nn.Linear(in_features=n_dim_query, out_features=n_dim_memory, bias=False)

        self._dim_query = n_dim_query
        self._dim_key = n_dim_memory

    def forward(self, source: Tensor, memory_bank: Tensor, mask: Optional[torch.Tensor] = None) -> (Tensor, Tensor):
        """
        simple self attention layer that has query transformation layer

        :param source: 2-D tensor (N_batch, N_dim_query). typically, output of RNN(=lstm, gru) state: (N_batch, N_dim_state)
        :param memory_bank: set of context vector: (N_batch, N_context, N_dim_memory)
        :param mask: binary mask indicating which keys should have non-zero attention. `(N_batch, N_context)`
        :return: 2-D context tensor (N_batch, N_dim_memory)
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

        if mask is not None:
            # mask: (N_mb, N_context) -> (N_mb, N_context, 1)
            mask = mask.unsqueeze(2)
            v_score = v_score.masked_fill(mask, -1e18)

        # v_attn = (N_batch, N_context, 1)
        v_attn = F.softmax(v_score, dim=1)

        # calculate context vector
        # v_context = (N_batch, N_dim_key), v_attn = (N_batch, N_context)
        v_context = torch.sum(h_t * v_attn, dim=1, keepdim=False)
        v_attn = v_attn.squeeze(dim=-1)

        return v_context, v_attn


class MultiHeadedAttention(nn.Module):
    """
    simple Multi-Head Attention module that is similar to "Attention is All You Need".

    Args:
       n_head (int): number of parallel heads
       n_dim_query (int): the dimension of queries
       n_dim_memory (int): the dimension of keys/values
       n_dim_out (int): the dimension of output: concatenation of the output of each heads. it must be divisible by n_head
       dropout (float): _dropout parameter
       transform_memory_bank (bool): allow transformation of keys/values or not.
    """

    def __init__(self, n_head: int, n_dim_query: int, n_dim_memory: int, n_dim_out: int, dropout=0.0, transform_memory_bank: bool = False):
        # assertion
        assert n_dim_out % n_head == 0
        if transform_memory_bank:
            warnings.warn("memory bank will be transformed using linear layer w/o bias.")
        else:
            assert n_dim_memory * n_head == n_dim_out, "if you don't transform memory bank, `n_dim_out` must be identical to `n_dim_memory*n_head`."

        self._n_dim_memory_per_head = n_dim_out // n_head
        self._n_dim_query = n_dim_query
        self._transform_memory_bank = transform_memory_bank

        super(MultiHeadedAttention, self).__init__()

        self._n_head = n_head

        # transformation layers
        # Q: (N_mb, N_query) -> (N_mb, N_out)
        # K: (N_mb, N_memory) -> (N_mb, N_out)
        # if transform_memory_bank:
        #   V: (N_mb, N_memory) -> (N_mb, N_out)
        # else:
        #   V: do not transform; (N_mb, N_head * N_memory = N_out)
        self._transform_queries = nn.Linear(n_dim_query, n_head * self._n_dim_memory_per_head, bias=False)
        if self._transform_memory_bank:
            self._transform_keys = nn.Linear(n_dim_memory, n_head * self._n_dim_memory_per_head, bias=False)
            self._transform_values = nn.Linear(n_dim_memory, n_head * self._n_dim_memory_per_head, bias=False)
        else:
            self._transform_keys = None
            self._transform_values = None

        self._dropout = nn.Dropout(dropout)

    def _split_into_heads(self, x):
        n_mb = x.size(0)
        return x.view(n_mb, -1, self._n_head, self._n_dim_memory_per_head).transpose(1,2)

    def _concat_heads_into_one(self, x):
        n_mb = x.size(0)
        return x.transpose(1,2).contiguous().view(n_mb,-1,self._n_head * self._n_dim_memory_per_head)

    def forward(self, source: torch.Tensor, memory_bank: torch.Tensor, mask: Optional[torch.Tensor] = None) -> (Tensor, Tensor):
        """
        Compute the context vector and the attention vectors.
        Args:
           source (`FloatTensor`): single query. `[batch, n_dim_query]`
           memory_bank (`FloatTensor`): set of `n_sample` memory_bank vectors `[batch, n_sample, n_dim_memory]`
           mask: binary mask indicating which keys should have non-zero attention. `[batch, n_sample]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[n_batch, n_dim_out]`
           * output attention vectors `[n_batch, n_head, n_sample]`
        """

        n_head = self._n_head

        # 1) Project source->query, memory_bank->key and value
        # query: (N_mb, N_query) -> (N_mb, N_out = N_head * N_memory_per_head)
        v_query = self._transform_queries(source)
        if self._transform_memory_bank:
            # key & value: (N_mb, N_sample, N_dim_memory) -> (N_mb, N_sample, N_out = N_head * N_memory_per_head)
            v_key = self._transform_keys(memory_bank)
            v_value = self._transform_values(memory_bank)
        else:
            # key & value: (N_mb, N_sample, N_dim_memory) -> (N_mb, N_sample, N_out = N_head * N_dim_memory)
            v_key = memory_bank.repeat(1,1,n_head)
            v_value = memory_bank.repeat(1,1,n_head)

        # split to each heads and transpose axes
        # query: (N_mb, N_out) -> (N_mb, N_head, 1, N_memory_per_head)
        # key & value: (N_mb, N_sample, N_out) -> (N_mb, N_head, N_sample, N_memory_per_head)
        v_query = self._split_into_heads(v_query)
        v_key = self._split_into_heads(v_key)
        v_value = self._split_into_heads(v_value)

        # 2) Calculate and scale scores.
        v_query = v_query / np.sqrt(self._n_dim_memory_per_head)
        # scores = (N_mb, N_head, 1, N_sample)
        # scores[b,h,0,s] = <v_query[b,h,0,:], v_key[b,h,s,:]>
        v_scores = torch.matmul(v_query, v_key.transpose(2,3))

        if mask is not None:
            # mask: (N_mb, N_sample) -> (N_mb, 1, 1, N_sample)
            mask = mask.unsqueeze(1).unsqueeze(1)
            v_scores = v_scores.masked_fill(mask, -1e18)

        # 3) Apply attention and dropout and compute context vectors.
        # attn = drop_attn = (N_mb, N_head, 1, N_sample)
        # context_h = (N_mb, N_head, 1, N_memory_per_head)
        # context_h[b,h,0,d] = <drop_attn[b,h,0,:], v_value[b,h,:,d]>
        # context = (N_mb, N_head * N_memory_per_head)
        attn = F.softmax(v_scores, dim=-1)
        drop_attn = self._dropout(attn)
        context_h = torch.matmul(drop_attn, v_value)
        context = self._concat_heads_into_one(context_h)

        # return squeezed version
        context = context.squeeze()
        attn = attn.squeeze()

        return context, attn