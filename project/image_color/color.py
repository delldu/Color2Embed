# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:46:28 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
from . import data

import math
import torch.nn.functional as F
from torchvision import models
from typing import List

import pdb


def vgg_normal(tensor):
    mean_val = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
    std_val = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
    tensor_norm = (tensor - mean_val) / std_val
    return tensor_norm


class VGG19Extractor(nn.Module):
    def __init__(self):
        super(VGG19Extractor, self).__init__()
        vgg_model = models.vgg19()
        vgg_feature = vgg_model.features
        seq_list = [ele for ele in vgg_feature]

        # conv1_1 = seq_list[0](x)
        # relu1_1 = seq_list[1](conv1_1)
        # conv1_2 = seq_list[2](relu1_1)
        # relu1_2 = seq_list[3](conv1_2)
        # pool1 = seq_list[4](relu1_2)

        # conv2_1 = seq_list[5](pool1)
        # relu2_1 = seq_list[6](conv2_1)
        # conv2_2 = seq_list[7](relu2_1)
        # relu2_2 = seq_list[8](conv2_2)
        # pool2 = seq_list[9](relu2_2)

        # conv3_1 = seq_list[10](pool2)
        # relu3_1 = seq_list[11](conv3_1)
        # conv3_2 = seq_list[12](relu3_1)
        # relu3_2 = seq_list[13](conv3_2)
        # conv3_3 = seq_list[14](relu3_2)
        # relu3_3 = seq_list[15](conv3_3)
        # conv3_4 = seq_list[16](relu3_3)
        # relu3_4 = seq_list[17](conv3_4)
        # pool3 = seq_list[18](relu3_4)

        # conv4_1 = seq_list[19](pool3)
        # relu4_1 = seq_list[20](conv4_1)
        # conv4_2 = seq_list[21](relu4_1)
        # relu4_2 = seq_list[22](conv4_2)
        # conv4_3 = seq_list[23](relu4_2)
        # relu4_3 = seq_list[24](conv4_3)
        # conv4_4 = seq_list[25](relu4_3)
        # relu4_4 = seq_list[26](conv4_4)
        # pool4 = seq_list[27](relu4_4)

        # conv5_1 = seq_list[28](pool4)
        # relu5_1 = seq_list[29](conv5_1)
        # conv5_2 = seq_list[30](relu5_1)
        # relu5_2 = seq_list[31](conv5_2) # [B, 512, 16, 16]

        # conv5_3 = seq_list[32](relu5_2)
        # relu5_3 = seq_list[33](conv5_3)
        # conv5_4 = seq_list[34](relu5_3)
        # relu5_4 = seq_list[35](conv5_4)
        # pool5 = seq_list[36](relu5_4) # [B, 512, 8, 8]

        self.block = nn.ModuleList()
        for i in range(0, 32):
            self.block.append(seq_list[i])

    def forward(self, x):
        """Return relu5_2 feature"""
        x = vgg_normal(x)
        for i, blk in enumerate(self.block):
            x = blk(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.bottle_conv(x)
        x = self.double_conv(x) + x
        return x / math.sqrt(2)


class Down(nn.Module):
    """Downscaling with stride conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            ResBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.main(x)


class SDFT(nn.Module):
    def __init__(self, color_dim, channels, kernel_size=3):
        super().__init__()

        # generate global conv weights
        fan_in = channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(color_dim, channels, 1)
        self.weight = nn.Parameter(torch.randn(1, channels, channels, kernel_size, kernel_size))

    def forward(self, fea, color_style):
        # for global adjustation
        B, C, H, W = fea.size()
        # print(fea.shape, color_style.shape)
        style = self.modulation(color_style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(B * C, C, self.kernel_size, self.kernel_size)

        fea = fea.view(1, B * C, H, W)
        fea = F.conv2d(fea, weight, padding=self.padding, groups=B)
        fea = fea.view(B, C, H, W)

        return fea


class UpBlock(nn.Module):
    def __init__(self, color_dim, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.conv_s = nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0)

        # generate global conv weights
        self.SDFT = SDFT(color_dim, out_channels, kernel_size)

    def forward(self, x1, x2, color_style):
        x1 = self.up(x1)
        x1_s = self.conv_s(x1)

        x = torch.cat([x1, x2[:, ::4, :, :]], dim=1)
        x = self.conv_cat(x)
        x = self.SDFT(x, color_style)

        x = x + x1_s

        return x


class ColorEncoder(nn.Module):
    def __init__(self, color_dim=512):
        super(ColorEncoder, self).__init__()
        self.color_dim = color_dim
        self.vgg = VGG19Extractor()
        self.feature2vector = nn.Sequential(
            nn.Conv2d(color_dim, color_dim, 4, 2, 2),  # 8x8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 4, 2, 2),  # 4x4
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            nn.Conv2d(color_dim, color_dim // 2, 1),  # linear-1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim // 2, color_dim // 2, 1),  # linear-2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(color_dim // 2, color_dim, 1),  # linear-3
        )

    def forward(self, x):
        # x is RGB tensor of reference image
        vgg_feature = self.vgg(x)  # [B, 512, 16, 16]
        color_vector = self.feature2vector(vgg_feature)  # [B, 512, 1, 1]
        # color_vector.size() -- [1, 512, 1, 1]
        return color_vector


class ColorDecoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=3):
        super(ColorDecoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2
        self.down4 = Down(512, 1024 // factor)
        self.up1 = UpBlock(512, 1024, 512 // factor, 3)
        self.up2 = UpBlock(512, 512, 256 // factor, 3)
        self.up3 = UpBlock(512, 256, 128 // factor, 5)
        self.up4 = UpBlock(512, 128, 64, 5)
        self.outc = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(64, 2, 3, 1, 1), nn.Tanh()
        )

    def forward(self, x: List[torch.Tensor]):
        # (Pdb) x[0].size() -- [1, 1, 256, 256]
        # (Pdb) x[1].size() -- [1, 512, 1, 1]
        gray_tensor = x[0]
        color_vector = x[1]  # [B, 512, 1, 1]

        x1 = self.inc(gray_tensor)  # [B, 64, 256, 256]
        x2 = self.down1(x1)  # [B, 128, 128, 128]
        x3 = self.down2(x2)  # [B, 256, 64, 64]
        x4 = self.down3(x3)  # [B, 512, 32, 32]
        x5 = self.down4(x4)  # [B, 512, 16, 16]

        x6 = self.up1(x5, x4, color_vector)  # [B, 256, 32, 32]
        x7 = self.up2(x6, x3, color_vector)  # [B, 128, 64, 64]
        x8 = self.up3(x7, x2, color_vector)  # [B, 64, 128, 128]
        x9 = self.up4(x8, x1, color_vector)  # [B, 64, 256, 256]
        x_ab = self.outc(x9)

        # x_ab.size() -- [1, 2, 256, 256]
        # x_ab.min(), x_ab.max() -- -0.2950, 0.4178

        return x_ab


class ColorModel(nn.Module):
    def __init__(self):
        super(ColorModel, self).__init__()
        self.encoder = ColorEncoder()
        self.decoder = ColorDecoder()

        # for smoke test
        self.max_h = 1024
        self.max_w = 2048
        self.max_times = 1
        self.scale = 1
        # GPU 2G, 40ms


    def forward(self, grey, color):
        # gray in [0, 1.0]
        H, W = int(grey.size(2)), int(grey.size(3))

        gray_lab = data.rgb2lab(grey)
        gray_256 = F.interpolate(
            gray_lab[:, 0:1, :, :],
            size=(256, 256),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )
        # gray_256 in L-space, [-1.0, 1.0]
        color_256 = F.interpolate(
            color,
            size=(256, 256),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )
        color_vector = self.encoder(color_256)
        fake_ab_256 = self.decoder([gray_256, color_vector])
        fake_ab = F.interpolate(
            fake_ab_256,
            size=(H, W),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )
        color_lab = torch.cat((gray_lab[:, 0:1, :, :], fake_ab), dim=1)

        return data.lab2rgb(color_lab)
