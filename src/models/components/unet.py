"""Based on the implementation of the UNet used by Facebook in the FastMRI challenge."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """2D Double Convolutional layer.

    Args:
    in_channels: number of input channels
    out_channels: number of output channels
    dropout_probability: probability of dropout after activation
    """

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_probability),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_probability),
        )

    def forward(self, input):
        """
        Args:
        input: input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        output: output tensor of shape (batch_size, out_channels, height, width)
        """
        return self.layers(input)


class UNet(nn.Module):
    """2D UNet architecture.

    Args:
    in_channels: number of input channels
    out_channels: number of output channels (number of classes)
    channels: number of channels in the first layer, which is doubled after each down sampling
    depth: number of down sampling and up sampling layers
    dropout_probability: probability of dropout after activation
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 16,
        depth: int = 4,
        dropout_probability: float = 0.0,
    ):
        super().__init__()

        self.depth = depth
        self.dropout_probability = dropout_probability

        self.down_sample_layers = nn.ModuleList(
            [DoubleConv(in_channels, channels, dropout_probability)]
        )

        for i in range(depth - 1):
            self.down_sample_layers += [DoubleConv(channels, channels * 2, dropout_probability)]
            channels *= 2

        self.conv = DoubleConv(channels, channels, dropout_probability)

        self.up_sample_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.up_sample_layers += [DoubleConv(channels * 2, channels // 2, dropout_probability)]
            channels //= 2
        self.up_sample_layers += [DoubleConv(channels * 2, channels, dropout_probability)]

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.Conv2d(channels // 2, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
        input: input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        output: output tensor of shape (batch_size, out_channels, height, width)
        out_channels is the number of classes. A Bernoulli variable is returned for each class.
        """
        stack = []
        output = input

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        output = self.conv2(output)

        return {"seg_logits": output}
