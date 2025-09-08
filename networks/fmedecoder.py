from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .PMSFE import *  # ConvBlock, Conv3x3, Attention_Module
from timm.models.layers import trunc_normal_
from .raem import RAEM
from .spm import SPM
from networks.sync_batchnorm import SynchronizedBatchNorm2d

bn_mom = 0.0003

# DF_Module 相关定义
class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


def depthwise_separable_conv(in_chn, out_chn):
    return nn.Sequential(
        # Depthwise Conv (计算量低)
        nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1, groups=in_chn),
        nn.ReLU(inplace=True),
        # Pointwise Conv (1×1 调整通道)
        nn.Conv2d(in_chn, out_chn, kernel_size=1),
        nn.ReLU(inplace=True),
    )


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        return self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))

class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                SynchronizedBatchNorm2d(dim_in//2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


# 主解码器类
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc  # 例如 [64, 128, 256, 512]
        self.num_ch_dec = (np.array(num_ch_enc) / 2).astype('int')  # 例如 [32, 64, 128, 256]

        self.convs = nn.ModuleDict()
        self.convs["f2"] = Attention_Module(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs["f0"] = Attention_Module(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs["spm"] = SPM(self.num_ch_enc[2])

        for i in range(2, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv_{}_0".format(i)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                # 添加 DF_Module 和通道调整
                self.convs[f"skip_adjust_{i}"] = nn.Conv2d(self.num_ch_enc[i - 1], num_ch_in, kernel_size=1)
                self.convs[f"dfm_{i}"] = DF_Module(dim_in=num_ch_in, dim_out=num_ch_in + self.num_ch_enc[i - 1], reduction=False)
                num_ch_in += self.num_ch_enc[i - 1]  # 与 torch.cat 输出一致
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"raem_{i}"] = RAEM(num_ch_in)
            self.convs["upconv_{}_1".format(i)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs["dispconv_{}".format(s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}

        feat = {}
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = self.convs["f0"](input_features[0])

        x = self.convs["spm"](feat[2])

        for i in range(2, -1, -1):
            x = self.convs["upconv_{}_0".format(i)](x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            if self.use_skips and i > 0:
                skip = self.convs[f"skip_adjust_{i}"](feat[i - 1])  # 调整跳跃连接通道
                x = self.convs[f"dfm_{i}"](x, skip)  # 使用 DF_Module 融合

            x = self.convs[f"raem_{i}"](x)
            x = self.convs["upconv_{}_1".format(i)](x)

            self.outputs[('d_feature', i)] = x

            if i in self.scales:
                f = self.convs["dispconv_{}".format(i)](x)
                f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
                self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs
