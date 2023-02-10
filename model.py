import torch
import torch.nn as nn
from utils import visualize
from torchvision import models


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
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)


class U_Net(nn.Module):
    """
    UNet
    """
    @property
    def is_transpose(self):
        return True

    @property
    def init_ch(self):
        return 16

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8, self.init_ch * 16]

    def __init__(self, in_channel=1, out_channel=1):
        super(U_Net, self).__init__()

        # Encoder
        self.Conv1 = DoubleConv(in_channel, self.channels[0])
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = DoubleConv(self.channels[0], self.channels[1])
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = DoubleConv(self.channels[1], self.channels[2])
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = DoubleConv(self.channels[2], self.channels[3])
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv5 = DoubleConv(self.channels[3], self.channels[4])

        # Decoder
        self.Up5 = Up(self.channels[4], self.channels[3], self.is_transpose)

        self.Up_conv4 = DoubleConv(self.channels[4], self.channels[3])
        self.Up4 = Up(self.channels[3], self.channels[2], self.is_transpose)

        self.Up_conv3 = DoubleConv(self.channels[3], self.channels[2])
        self.Up3 = Up(self.channels[2], self.channels[1], self.is_transpose)

        self.Up_conv2 = DoubleConv(self.channels[2], self.channels[1])
        self.Up2 = Up(self.channels[1], self.channels[0], self.is_transpose)

        self.Up_conv1 = DoubleConv(self.channels[1], self.channels[0])
        self.Out = nn.Conv2d(self.channels[0], out_channel, kernel_size=1, stride=1, padding=0)

    # def cut(self, x):
    #     self.bs = x.shape[0]
    #     self.w, self.h = int(x.shape[2]/2), int(x.shape[3]/2)
    #     img = torch.zeros((self.bs*4, x.shape[1], self.w, self.h), device=x.device, dtype=torch.float32)
    #     img[:self.bs, ...] = x[..., :self.w, :self.h]
    #     img[self.bs:2*self.bs, ...] = x[..., self.w:2*self.w, :self.h]
    #     img[2*self.bs:3*self.bs, ...] = x[..., :self.w, self.h:2*self.h]
    #     img[3*self.bs:4*self.bs, ...] = x[..., self.w:2*self.w, self.h:2*self.h]
    #
    #     return img
    #
    # def merge(self, x):
    #     img = torch.zeros((self.bs, x.shape[1], 2 * self.w, 2 * self.h), device=x.device, dtype=torch.float32)
    #     img[..., :self.w, :self.h] = x[:self.bs, ...]
    #     img[..., self.w:2*self.w, :self.h] = x[self.bs:2*self.bs, ...]
    #     img[..., :self.w, self.h:2*self.h] = x[2*self.bs:3*self.bs, ...]
    #     img[..., self.w:2*self.w, self.h:2*self.h] = x[3*self.bs:4*self.bs, ...]
    #
    #     return img

    def forward(self, x):
        # im1 = x['im1']
        # im2 = x['im2']
        # x = torch.cat((im1, im2), dim=1)
        # x = im2
        # x = self.cut(x)
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5)

        # Decoder
        d4 = self.Up5(e5)

        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up4(d4)

        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up3(d3)

        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Up2(d2)

        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.Up_conv1(d1)
        out = self.Out(d1)

        # out = self.merge(out)

        return out


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


class DetModel(nn.Module):
    """
    UNet
    """

    @property
    def is_transpose(self):
        return False

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


class U_Net_down3(nn.Module):
    """
    UNet
    """
    @property
    def is_transpose(self):
        return False

    @property
    def init_ch(self):
        return 16

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8]

    def __init__(self, in_channel=1, out_channel=1):
        super(U_Net_down3, self).__init__()

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = x['im2']
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


class U_Net_resnet18(nn.Module):
    """
    UNet
    """
    @property
    def is_transpose(self):
        return False

    @property
    def init_ch(self):
        return 64

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8, self.init_ch * 16]

    def __init__(self, in_channel=1, out_channel=1):
        super(U_Net_resnet18, self).__init__()

        # Encoder
        resnet = models.resnet18(pretrained=False)

        self.firstconv = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.Up5 = Up(self.channels[3], self.channels[2], self.is_transpose)

        self.Up_conv4 = DoubleConv(self.channels[3], self.channels[2])
        self.Up4 = Up(self.channels[2], self.channels[1], self.is_transpose)

        self.Up_conv3 = DoubleConv(self.channels[2], self.channels[1])
        self.Up3 = Up(self.channels[1], self.channels[0], self.is_transpose)

        self.Up_conv2 = DoubleConv(self.channels[1], self.channels[0])
        self.Up2 = Up(self.channels[0], self.channels[0], self.is_transpose)

        self.Up_conv1 = DoubleConv(self.channels[1], self.channels[0])
        self.Out = nn.Conv2d(self.channels[0], out_channel, kernel_size=1, stride=1, padding=0)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad


    def forward(self, x):
        # im1 = x['im1']
        # im2 = x['im2']
        # x = torch.cat((im1, im2), dim=1)
        # x = im2
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        e1 = self.firstrelu(x)
        e2 = self.firstmaxpool(e1)

        e2 = self.encoder1(e2)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)

        # Decoder
        d4 = self.Up5(e5)

        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up4(d4)

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


class U_Net_vgg16(nn.Module):
    """
    UNet
    """
    @property
    def is_transpose(self):
        return False

    @property
    def init_ch(self):
        return 64

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8, self.init_ch * 16]

    def __init__(self, in_channel=1, out_channel=1):
        super(U_Net_vgg16, self).__init__()

        # Encoder
        vgg = models.vgg16_bn(pretrained=True)

        features = list(vgg.features)[:44]

        self.features1 = list(features)[:6]
        self.features1[0] = nn.Conv2d(in_channel, self.channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.features2 = list(features)[6:13]
        self.features3 = list(features)[13:23]
        self.features4 = list(features)[23:33]
        self.features5 = list(features)[33:43]

        self.features1 = nn.Sequential(*self.features1)
        self.features2 = nn.Sequential(*self.features2)
        self.features3 = nn.Sequential(*self.features3)
        self.features4 = nn.Sequential(*self.features4)
        self.features5 = nn.Sequential(*self.features5)

        # Decoder
        self.Up5 = Up(self.channels[3], self.channels[3], self.is_transpose)

        self.Up_conv4 = DoubleConv(self.channels[4], self.channels[3])
        self.Up4 = Up(self.channels[3], self.channels[2], self.is_transpose)

        self.Up_conv3 = DoubleConv(self.channels[3], self.channels[2])
        self.Up3 = Up(self.channels[2], self.channels[1], self.is_transpose)

        self.Up_conv2 = DoubleConv(self.channels[2], self.channels[1])
        self.Up2 = Up(self.channels[1], self.channels[0], self.is_transpose)

        self.Up_conv1 = DoubleConv(self.channels[1], self.channels[0])
        self.Out = nn.Conv2d(self.channels[0], out_channel, kernel_size=1, stride=1, padding=0)

    def require_encoder_grad(self, requires_grad):
        blocks = [
                  self.features1,
                  self.features2,
                  self.features3,
                  self.features4,
                  self.features5]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def check(self, block):
        for i, p in enumerate(block.parameters()):
            if i == 0:
                print(p)
                print(p.requires_grad)

    def forward(self, x):
        # im1 = x['im1']
        im2 = x['im2']
        # x = torch.cat((im1, im2), dim=1)
        x = im2
        # Encoder
        e1 = self.features1(x)
        e2 = self.features2(e1)
        e3 = self.features3(e2)
        e4 = self.features4(e3)

        e5 = self.features5(e4)

        # Decoder
        d4 = self.Up5(e5)

        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up4(d4)

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


class U_Net_resnet(nn.Module):
    """
    UNet
    """

    @property
    def is_transpose(self):
        return False

    @property
    def init_ch(self):
        return 16

    @property
    def channels(self):
        return [self.init_ch, self.init_ch * 2, self.init_ch * 4, self.init_ch * 8, self.init_ch * 16]

    def __init__(self, in_channel=1, out_channel=1):
        super(U_Net_resnet, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, self.channels[0], 3, 1, 1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.layer2 = BLock_Layer(BasicBlock, self.channels[0], self.channels[0], 2, False)
        self.layer3 = BLock_Layer(BasicBlock, self.channels[0], self.channels[1], 2, True)
        self.layer4 = BLock_Layer(BasicBlock, self.channels[1], self.channels[2], 2, True)
        self.layer5 = BLock_Layer(BasicBlock, self.channels[2], self.channels[3], 2, True)

        # Decoder
        self.Up5 = Up(self.channels[3], self.channels[2], self.is_transpose)

        # self.Up_conv4 = DoubleConv(self.channels[3], self.channels[2])
        self.Up_conv4 = BLock_Layer(BasicBlock, self.channels[3], self.channels[2], 2, False)
        self.Up4 = Up(self.channels[2], self.channels[1], self.is_transpose)

        # self.Up_conv3 = DoubleConv(self.channels[2], self.channels[1])
        self.Up_conv3 = BLock_Layer(BasicBlock, self.channels[2], self.channels[1], 2, False)
        self.Up3 = Up(self.channels[1], self.channels[0], self.is_transpose)

        # self.Up_conv2 = DoubleConv(self.channels[1], self.channels[0])
        self.Up_conv2 = BLock_Layer(BasicBlock, self.channels[1], self.channels[0], 2, False)
        # self.Up2 = Up(self.channels[0], self.channels[0], self.is_transpose)

        # self.Up_conv1 = DoubleConv(self.channels[1], self.channels[0])
        self.Out = nn.Conv2d(self.channels[0], out_channel, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # im1 = x['im1']
        # im2 = x['im2']
        # x = torch.cat((im1, im2), dim=1)
        # x = im2

        x = self.conv1(x)
        e2 = self.layer2(x)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)

        # Decoder
        d4 = self.Up5(e5)

        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up4(d4)

        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up3(d3)

        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Out(d2)

        return out


class BLock_Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_block, downsample):
        super(BLock_Layer, self).__init__()
        layers = []
        if downsample:
            layers.append(block(in_planes, planes, 2))
        else:
            layers.append(block(in_planes, planes, 1))
        for _ in range(1, num_block):
            layers.append(block(planes, planes, 1))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.shortcut(x) + self.residual(x))
