from math import floor
import torch.nn as nn

__all__ = ['mobilenet_tiny_imagenet']


class MobileNet_tiny_imagenet(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super(MobileNet_tiny_imagenet, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            # ofm size: 64x32x32
            depthwise_conv(self.channels[1], self.channels[2], 2),
            # ofm size: 128x16x16
            depthwise_conv(self.channels[2], self.channels[2], 1),
            # ofm size: 128x16x16
            depthwise_conv(self.channels[2], self.channels[3], 2),
            # ofm size: 256x8x8
            depthwise_conv(self.channels[3], self.channels[3], 1),
            # ofm size: 256x8x8
            depthwise_conv(self.channels[3], self.channels[4], 2),
            # ofm size: 512x4x4
            depthwise_conv(self.channels[4], self.channels[4], 1),
            # ofm size: 512x4x4
            depthwise_conv(self.channels[4], self.channels[4], 1),
            # ofm size: 512x4x4
            depthwise_conv(self.channels[4], self.channels[4], 1),
            # ofm size: 512x4x4
            depthwise_conv(self.channels[4], self.channels[4], 1),
            # ofm size: 512x4x4
            depthwise_conv(self.channels[4], self.channels[4], 1),
            # ofm size: 512x4x4
            depthwise_conv(self.channels[4], self.channels[5], 2),
            # ofm size: 1024x2x2
            depthwise_conv(self.channels[5], self.channels[5], 1),
            # ofm size: 1024x2x2
            nn.AvgPool2d(2),
            # ofm size: 1024x1x1
        )
        self.fc = nn.Linear(self.channels[5], 200, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x

def mobilenet_tiny_imagenet():
    return MobileNet_tiny_imagenet()
