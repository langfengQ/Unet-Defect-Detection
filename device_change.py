from model import DetModel
import torch

device = 'cuda:0'
model = DetModel(in_channel=1, out_channel=1).to(device)
pretrained_model = torch.load('./70_final_vgg16ch24_d3_olytrainremove_08_bilinear.pth', map_location=device)
model.load_state_dict(pretrained_model['model'])
torch.save(model.state_dict(), './Model_parameter.p')
print('finish')