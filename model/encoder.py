#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from .multi_layer import MultiDenseLayer
from typing import Optional

class LSTMEncoder(nn.Module):

    __pad_index = 0
    __batch_first = True

    def __init__(self, n_vocab, n_dim_embedding, n_dim_lstm_hidden, n_lstm_layer, bidirectional,
                 embedding_layer: Optional[nn.Embedding] = None,
                 highway=False, return_state=False, device=torch.device("cpu"), **kwargs):

        super(__class__, self).__init__()

        self._n_vocab = n_vocab
        if embedding_layer is not None:
            self._n_dim_embedding = embedding_layer.embedding_dim
        else:
            self._n_dim_embedding = n_dim_embedding
        self._n_dim_lstm_hidden = n_dim_lstm_hidden
        self._n_lstm_layer = n_lstm_layer
        self._bidirectional = bidirectional
        self._highway = highway
        self._return_state = return_state
        self._device = device

        if embedding_layer is not None:
            self._embed = embedding_layer
        else:
            self._embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim_embedding, padding_idx=self.__pad_index)
        self._lstm = nn.LSTM(input_size=n_dim_embedding, hidden_size=n_dim_lstm_hidden, num_layers=n_lstm_layer,
                             batch_first=self.__batch_first, bidirectional=bidirectional)

    def _init_state(self, batch_size):
        # zero init
        n_dim = self._n_lstm_layer*2 if self._bidirectional else self._n_lstm_layer
        h_0 = torch.zeros(n_dim, batch_size, self._n_dim_embedding, device=self._device)
        c_0 = torch.zeros(n_dim, batch_size, self._n_dim_embedding, device=self._device)
        return h_0, c_0

    def forward(self, x_seq, seq_len):
        """

        :param x_seq: batch of sequence of index; torch.tensor((n_mb, n_seq_len_max), dtype=torch.long)
        :param seq_len: batch of sequence length; torch.tensor((n_mb,), dtype=torch.long)

        return: (encoded sequence, sequence length, optional[final state(h_n,c_n)])
        """
        batch_size, n_seq_len_max = x_seq.size()

        # (n_mb, n_seq_len_max, n_dim_embedding)
        embed = self._embed(x_seq)
        # ignore padded state
        v = nn.utils.rnn.pack_padded_sequence(embed, lengths=seq_len, batch_first=self.__batch_first)

        h_0_c_0 = self._init_state(batch_size=batch_size)
        h, h_n_c_n = self._lstm(v, h_0_c_0)

        # undo the packing operation
        h, v_seq_len = nn.utils.rnn.pad_packed_sequence(h, batch_first=self.__batch_first, padding_value=0., total_length=n_seq_len_max)

        if self._highway:
            h = torch.cat([h,embed], dim=-1)

        if self._return_state:
            return h, seq_len, h_n_c_n
        else:
            return h, seq_len


class GMMLSTMEncoder(LSTMEncoder):

    __pad_index = 0
    __batch_first = True

    def __init__(self, n_vocab: int, n_dim_embedding: int, n_dim_lstm_hidden: int, n_lstm_layer: int, bidirectional: bool,
                 encoder_alpha: MultiDenseLayer, encoder_mu: MultiDenseLayer, encoder_sigma: MultiDenseLayer,
                 custom_embedding_layer: Optional[nn.Embedding] = None,
                 highway: bool=False, apply_softmax: bool=True, return_state: bool=False, device=torch.device("cpu"), **kwargs):

        super(__class__, self).__init__(n_vocab, n_dim_embedding, n_dim_lstm_hidden, n_lstm_layer, bidirectional,
                                        custom_embedding_layer, highway, return_state, device, **kwargs)
        self._enc_alpha = encoder_alpha
        self._enc_mu = encoder_mu
        self._enc_sigma = encoder_sigma
        self._apply_softmax = apply_softmax

    def _masked_softmax(self, x: torch.Tensor, mask: torch.Tensor, dim=1):
        masked_vec = x * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def forward(self, x_seq, seq_len):

        # mask = (N_b, N_t)
        mask = (x_seq > self.__pad_index)

        # h = (N_b, N_t, N_h), seq_len = (N_b,)
        if self._return_state:
            h, seq_len, h_n_c_n = super(__class__, self).forward(x_seq, seq_len)
        else:
            h, seq_len = super(__class__, self).forward(x_seq, seq_len)
            h_n_c_n = None

        # z_alpha = (N_b, N_t)
        # alpha = (N_b, N_t)
        # alpha[b,:] = softmax(MLP(h[b,:]))
        z_alpha = self._enc_alpha(h)
        z_alpha = z_alpha.squeeze(dim=-1)
        if self._apply_softmax:
            alpha = self._masked_softmax(z_alpha, mask, dim=1)
        else:
            alpha = z_alpha * mask.float()

        # mu = (N_b, N_t, N_d)
        # mu[b,t] = MLP(h[b,t])
        z_mu = self._enc_mu(h)
        mu = z_mu * mask.float().unsqueeze(dim=-1)

        # sigma = (N_b, N_t)
        # sigma[b,t] = exp(MLP(h[b,t]))
        z_sigma = self._enc_sigma(h)
        z_sigma = z_sigma.squeeze(dim=-1)
        if z_sigma.ndimension() == 2:
            sigma = torch.exp(z_sigma) * mask.float()
        elif z_sigma.ndimension() == 3:
            sigma = torch.exp(z_sigma) * mask.float().unsqueeze(dim=-1)
        else:
            raise AttributeError("unsupported dimension size: %d" % z_sigma.ndimension())

        if self._return_state:
            return alpha, mu, sigma, h_n_c_n
        else:
            return alpha, mu, sigma
