from torch import Tensor
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            latent_vec_length: int = 100,
            output_shape: int = 64
    ) -> None:
        super(Generator, self).__init__()

        self.features = nn.Sequential(
            G_block(
                in_channels=100,
                out_channels=output_shape * 8,
                stride=1,
                padding=0),
            # Output: (64 * 8, 4, 4)

            G_block(
                in_channels=output_shape * 8,
                out_channels=output_shape * 4),
            # Output: (64 * 4, 8, 8)

            G_block(
                in_channels=output_shape * 4,
                out_channels=output_shape * 2),
            # Output: (64 * 2, 16, 16)

            G_block(
                in_channels=output_shape * 2,
                out_channels=output_shape),
            # Output: (64, 32, 32)

            nn.ConvTranspose2d(
                in_channels=output_shape,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False)
        )

        self.activation = nn.Tanh()
        # Output: (3, 64, 64)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.activation(x)
        return x


class G_block(nn.Module):
    def __init__(
            self,
            out_channels,
            in_channels: int = 3,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1
    ) -> None:
        super(G_block, self).__init__()
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d_trans(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            input_shape: int = 64
    ) -> None:
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            D_block(input_shape),
            # Output: (64, 32, 32)

            D_block(
                in_channels=input_shape,
                out_channels=input_shape * 2),
            # Output: (64 * 2, 16, 16)

            D_block(
                in_channels=input_shape * 2,
                out_channels=input_shape * 4),
            # Output: (64 * 4, 8, 8)

            D_block(
                in_channels=input_shape * 4,
                out_channels=input_shape * 8)
            # Output: (64 * 8, 4, 4)
        )

        self.classifier = nn.Conv2d(
            in_channels=input_shape * 8,
            out_channels=1,
            kernel_size=4,
            bias=False)
        # Output: (1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class D_block(nn.Module):
    def __init__(
            self,
            out_channels,
            in_channels: int = 3,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            alpha: float = 0.2
    ) -> None:
        super(D_block, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
