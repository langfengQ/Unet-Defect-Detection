
import torch.nn as nn


# conv_bn为网络的第一个卷积块，步长为2
def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# conv_dw为深度可分离卷积
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        # 3x3卷积提取特征，步长为2
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # 1x1卷积，步长为1
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):

    def __init__(self, n_channels, channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
            conv_bn(n_channels, channels[0], 1),

            conv_dw(channels[0], channels[1], 1),

            conv_dw(channels[1], channels[2], 2),
            conv_dw(channels[2], channels[2], 1),

            conv_dw(channels[2], channels[3], 2),
            conv_dw(channels[3], channels[3], 1),
        )
        self.layer2 = nn.Sequential(
            conv_dw(channels[3], channels[4], 2),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
            conv_dw(channels[4], channels[4], 1),
        )
        self.layer3 = nn.Sequential(
            conv_dw(channels[4], channels[5], 2),
            conv_dw(channels[5], channels[5], 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[5], 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)

        x = x.view(-1, 1024)
        x = self.fc(x)
        return x