import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    DoubleConv Block
    """

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """
    DoubleConv Block
    """

    def __init__(self, in_channel, out_channel):
        super(TripleConv, self).__init__()

        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.triple_conv(x)


class Up(nn.Module):
    """
    Up Block
    """

    def __init__(self, in_channel, out_channel, transpose=False):
        super(Up, self).__init__()
        if transpose:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)


class DetModel(nn.Module):
    """
    UNet
    """

    @property
    def is_transpose(self):
        return True

    @property
    def init_ch(self):
        return 24

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8]

    def __init__(self, in_channel=1, out_channel=1):
        super(DetModel, self).__init__()

        # Encoder
        self.Conv1 = DoubleConv(in_channel, self.channels[0])
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = DoubleConv(self.channels[0], self.channels[1])
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = TripleConv(self.channels[1], self.channels[2])
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = TripleConv(self.channels[2], self.channels[3])

        # Decoder
        self.Up4 = Up(self.channels[3], self.channels[2], self.is_transpose)

        self.Up_conv3 = TripleConv(self.channels[3], self.channels[2])
        self.Up3 = Up(self.channels[2], self.channels[1], self.is_transpose)

        self.Up_conv2 = DoubleConv(self.channels[2], self.channels[1])
        self.Up2 = Up(self.channels[1], self.channels[0], self.is_transpose)

        self.Up_conv1 = DoubleConv(self.channels[1], self.channels[0])
        self.Out = nn.Conv2d(self.channels[0], out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.center(e4)

        # Decoder
        d3 = self.Up4(e4)

        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up3(d3)

        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Up2(d2)

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.Up_conv1(d1)
        out = self.Out(d1)

        return out

    def predict(self, x):
        output = torch.sigmoid(self.forward((x['im2'] / 255. - 0.518) / 0.361))
        output[output > 0.1] = 1
        output[output <= 0.1] = 0

        return output