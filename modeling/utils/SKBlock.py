import torch.nn as nn
import torch
from functools import reduce


class SKConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        batch_size = x.size(0)
        output = []
        output.append(x)
        output.append(y)
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V
