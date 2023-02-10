from mobilenet import MobileNet
import torch.nn as nn
from collections import OrderedDict
import torch
import torchsummary as summary

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class mobilenet(nn.Module):
    def __init__(self, n_channels, channels):
        super(mobilenet, self).__init__()
        self.model = MobileNet(n_channels, channels)

    def forward(self, x):
        out3 = self.model.layer1(x)
        out4 = self.model.layer2(out3)
        out5 = self.model.layer3(out4)

        return out3, out4, out5


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    @property
    def init_ch(self):
        return 16

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8, self.init_ch * 16, self.init_ch * 32]

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        self.n_channels = in_channel
        self.num_classes = out_channel

        self.backbone = mobilenet(in_channel, channels=self.channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(self.channels[5], self.channels[4])

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(self.channels[5], self.channels[3])

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(self.channels[4], self.channels[2])

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = DoubleConv(self.channels[2], self.channels[1])

        self.oup = nn.Conv2d(self.channels[1], out_channel, kernel_size=1)

    def forward(self, x):
        #  backbone
        # x = x['im2']
        x2, x1, x0 = self.backbone(x)
        # print(f"x2.shape: {x2.shape}, x1: {x1.shape}, x0: {x0.shape} ")

        P5 = self.up1(x0)
        P5 = self.conv1(P5)           # P5: 26x26x512
        # print(P5.shape)
        P4 = x1                       # P4: 26x26x512
        P4 = torch.cat([P4, P5], dim=1)   # P4(堆叠后): 26x26x1024
        # print(f"cat 后是： {P4.shape}")

        P4 = self.up2(P4)             # 52x52x1024
        P4 = self.conv2(P4)           # 52x52x256
        P3 = x2                       # x2 = 52x52x256
        P3 = torch.cat([P4, P3], dim=1)  # 52x52x512

        P3 = self.up3(P3)
        P3 = self.conv3(P3)

        P3 = self.up4(P3)
        P3 = self.conv4(P3)

        out = self.oup(P3)
        # print(f"out.shape is {out.shape}")

        return out