#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.utils

### YOUR CODE HERE for part 1d

class Highway(nn.Module):
    """

    """
    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(embed_size, embed_size, bias=True)
        self.w_gate = nn.Linear(embed_size, embed_size, bias=True)
        self.rl = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x_conv_out):
        x_proj = self.w_proj(x_conv_out)
        x_proj = self.rl(x_proj)
        x_gate = self.w_gate(x_proj)
        x_gate = self.sig(x_gate)
        x_highway = x_proj*x_gate + (torch.ones(x_gate.size()) - x_gate) * x_conv_out
        return x_highway


### END YOUR CODE 

