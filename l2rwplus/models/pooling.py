import torch
from torch import nn


class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x**p, dim=-1) + eps) ** (1 / p)


class TemporalMeanPooling(nn.Module):
    def __init__(self, feat_dim=2048):
        super(TemporalMeanPooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        """
        :param x: shape [b,t,c]
        :return: shape [b,c]
        """
        b, t, c = x.size()

        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = x.view(b, -1)
        return x
