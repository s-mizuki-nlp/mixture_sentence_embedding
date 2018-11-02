#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from typing import Optional

class VariationalAutoEncoder(nn.Module):

    def __init__(self):

        super(__class__, self).__init__()

    def forward(self, x_seq: torch.Tensor, x_seq_len: torch.Tensor):
        pass