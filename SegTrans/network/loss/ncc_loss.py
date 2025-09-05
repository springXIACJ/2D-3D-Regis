import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


class NormalizedCrossCorrelation(nn.Module):

    def __init__(self, eps=1e-6, n_channels=1):
        super().__init__()
        self.eps = eps
        self._n_channels = n_channels

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def _normalized_cross_correlation(self, x, y) -> torch.Tensor:
        b, c, w, h = x.shape

        # Reshape
        x = x.view(b, c, -1)
        y = y.view(b, c, -1)

        # Mean
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        y_mean = torch.mean(y, dim=-1, keepdim=True)

        # Deviation
        x = x - x_mean
        y = y - y_mean

        dev_xy = torch.mul(x, y)
        dev_xx = torch.mul(x, x)
        dev_yy = torch.mul(y, y)

        dev_xx_sum = torch.sum(dev_xx, dim=-1, keepdim=True)
        dev_yy_sum = torch.sum(dev_yy, dim=-1, keepdim=True)

        ncc = torch.div(dev_xy, torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum) + self.eps))
        ncc = torch.sum(ncc, dim=-1, keepdim=True)
        return ncc.mean()

    def forward(self, x, y) -> torch.Tensor:
        self._check_input_dim(x)
        self._check_input_dim(y)
        ncc = self._normalized_cross_correlation(x, y)
        return 1. - ncc