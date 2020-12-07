from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__slots__ = ["MiniInceptionNet"]

from torch import Tensor


class MiniInceptionNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            use_separable: bool = False
    ) -> None:
        super().__init__()

        # first part - basic convolution
        self.first_conv = CustomConv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=1,
            separable=False)

        # second part - 2 inceptions and downsample
        self.second_inception_1 = Inception(
            in_channels=96,
            num_1x1_filters=32,
            num_3x3_filters=32,
            separable=use_separable)
        self.second_inception_2 = Inception(
            in_channels=64,
            num_1x1_filters=32,
            num_3x3_filters=48,
            separable=use_separable)
        self.second_downsample = Downsample(
            conv_in_channels=80,
            conv_out_channels=28)

        # third part - 4 inceptions and downsample
        self.third_inception_1 = Inception(
            in_channels=108,
            num_1x1_filters=112,
            num_3x3_filters=48,
            separable=use_separable)
        self.third_inception_2 = Inception(
            in_channels=160,
            num_1x1_filters=96,
            num_3x3_filters=64,
            separable=use_separable)
        self.third_inception_3 = Inception(
            in_channels=160,
            num_1x1_filters=112,
            num_3x3_filters=48,
            separable=use_separable)
        self.third_inception_4 = Inception(
            in_channels=160,
            num_1x1_filters=112,
            num_3x3_filters=48,
            separable=use_separable)
        self.third_downsample = Downsample(
            conv_in_channels=160,
            conv_out_channels=96)

        # fourth part - 2 inceptions, global average pooling and dropout
        self.fourth_inception_1 = Inception(
            in_channels=256,
            num_1x1_filters=176,
            num_3x3_filters=160,
            separable=use_separable)
        self.fourth_inception_2 = Inception(
            in_channels=336,
            num_1x1_filters=176,
            num_3x3_filters=160,
            separable=use_separable)
        # GAP, functional used in forward to take arbitrary shape
        self.fourth_dropout = nn.Dropout(p=0.5)

        # dense feature classifier
        self.fc_flatten = nn.Flatten()
        self.fc_linear = nn.Linear(
            in_features=336,
            out_features=num_classes)
        self.fc_softmax = nn.Softmax(dim=1)

        self.gradients = None

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        x = self.first_conv(x)

        x = self.second_inception_1(x)
        x = self.second_inception_2(x)
        x = self.second_downsample(x)

        x = self.third_inception_1(x)
        x = self.third_inception_2(x)
        x = self.third_inception_3(x)
        x = self.third_inception_4(x)
        x = self.third_downsample(x)

        x = self.fourth_inception_1(x)
        x = self.fourth_inception_2(x)

        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = self.fourth_dropout(x)

        x = self.fc_flatten(x)
        x = self.fc_linear(x)
        x = self.fc_softmax(x)

        return x


class CustomConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = None,
            padding: int = 0,
            separable: bool = False
    ) -> None:
        super(CustomConv2d, self).__init__()
        stride = stride if stride is not None else kernel_size
        self.separable = separable

        if not separable:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels)

            self.pointwise = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if not self.separable:
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_1x1_filters: int,
            num_3x3_filters: int,
            num_5x5_filters: int = 0,
            separable: bool = False,
    ) -> None:
        super(Inception, self).__init__()
        self.conv_1x1 = CustomConv2d(
            in_channels=in_channels,
            out_channels=num_1x1_filters,
            kernel_size=1,
            stride=1,
            separable=separable)

        self.conv_3x3 = CustomConv2d(
            in_channels=in_channels,
            out_channels=num_3x3_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            separable=separable)

        self.num_5x5_filters = num_5x5_filters
        if self.num_5x5_filters > 0:
            self.conv_5x5 = CustomConv2d(
                in_channels=in_channels,
                out_channels=num_5x5_filters,
                kernel_size=5,
                stride=1,
                padding=2,
                separable=separable)

    def forward(self, x: Tensor) -> Tensor:
        conv1_out = self.conv_1x1(x)
        conv3_out = self.conv_3x3(x)
        # concatenate along the channel dimension
        # (num_samples, channel, height, width)
        
        if self.num_5x5_filters == 0:
            x = torch.cat([conv1_out, conv3_out], dim=1)
        else:
            conv5_out = self.conv_5x5(x)
            x = torch.cat([conv1_out, conv3_out, conv5_out], dim=1)
        return x


class Downsample(nn.Module):
    def __init__(
            self,
            conv_in_channels: int,
            conv_out_channels: int
    ) -> None:
        super(Downsample, self).__init__()
        self.conv_3x3 = CustomConv2d(
            in_channels=conv_in_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2)

    def forward(self, x: Tensor) -> Tensor:
        conv_3x3 = self.conv_3x3(x)
        pool = self.pool(x)
        # concatenate along the channel dimension
        # (num_samples, channel, height, width)
        x = torch.cat([conv_3x3, pool], dim=1)
        return x
