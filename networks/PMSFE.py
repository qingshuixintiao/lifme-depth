from __future__ import absolute_import, division, print_function

import numpy as np
import math

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
        print(f"x type: {type(x)}")
        print(f"x shape: {x.shape if isinstance(x, torch.Tensor) else 'Not a Tensor'}")
## ChannelAttetion

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)

        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature
class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, output_channel=None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features):
        features = high_features

        features = self.ca(features)


        return self.relu(self.conv_se(features))


