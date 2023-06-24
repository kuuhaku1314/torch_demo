"""开发自己的层要继承nn.Module，一般需要重写__init__和forward方法"""
import torch
from torch import nn
from torch.nn import functional as F


class MySequential(nn.Module):

    def __init__(self, *args):
        super().__init__()
        for i, module in enumerate(args):
            self._modules[str(i)] = module

    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
