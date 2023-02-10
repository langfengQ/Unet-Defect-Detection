from tensorboardX import SummaryWriter
from model import *
from mobileUnet import UNet
import torch


input = torch.rand((2, 1, 512, 512))
writer = SummaryWriter('./nets/mobilenet')
model = UNet(in_channel=1, out_channel=1)
writer.add_graph(model, input_to_model=input, verbose=False)