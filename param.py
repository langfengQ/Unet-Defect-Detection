from model import *
import torch
import thop
from torchsummary import summary

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    from torchstat import stat
    device = torch.device("cuda:0")
    model1 = U_Net().to(device)
    # print(get_parameter_number(model1))
    input = torch.randn([4, 1, 32, 32], device=device)
    flops, params = thop.profile(model1, inputs=(input,))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops)
    print(params)