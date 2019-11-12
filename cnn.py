#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e

import torch
import torch.nn as nn
import torch.nn.utils

class CNN(nn.Module):
    """

    """
    def __init__(self, embed_size, f, kernel_size):
        super(CNN, self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.f = f
        self.conv = nn.Conv1d(embed_size, f, kernel_size)
        self.rl = nn.ReLU()
        # self.mp = nn.MaxPool1d(kernel_size)

    def forward(self, x_conv):
        x_conv_out = self.conv(x_conv)
        x_conv_out = self.rl(x_conv_out)
        # x_conv_out = self.mp(x_conv_out)
        x_conv_out, _ = torch.max(x_conv_out, 2)

        return x_conv_out


### END YOUR CODE

