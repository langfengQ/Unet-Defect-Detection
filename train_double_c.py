import os
import argparse
import numpy as np
import time
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import albumentations as A

from model import *
from Dataloader import My_Dataloader, data_label_list, InsertDataset
from utils import *
from data_prepare import *
from final_test import get_iou


def get_args():
    parser = argparse.ArgumentParser(description='trian')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--batch-size-test', type=int, default=20, help='batch size for testing')
    parser.add_argument('--total-epochs', type=int, default=70, help='number of epochs to train')
    # parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--use-cuda', action='store_false', help='use CUDA')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='CUDA number')
    parser.add_argument('--seed', type=int, default=24, help='random seed')
    parser.add_argument('--is-k-fold', action='store_false', help='')
    parser.add_argument('--k', type=int, default=10, help='k of k_fold method')

    parser.add_argument('--save', action='store_false', help='save model')
    parser.add_argument('--tensorboard', action='store_false', help='write tensorboard')
    parser.add_argument('--tensorboard-path', type=str, default='./summaries/final_vgg16ch16_d3_olytrainremove_08_bilinear/',
                        help='path of tensorboard')
    parser.add_argument('--save-model-interval', type=int, default=5,
                        help='save model every save_model_interval')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint/final_vgg16ch16_d3_olytrainremove_08_bilinear.p',
                        help='path of saved model')
    # parser.add_argument('--log-interval', type=int, default=20,
    #                     help='how many batches to wait before logging training status')

    args = parser.parse_args()
    return args


# 设置种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_load(args, k_index=1):
    data_path = os.getcwd() + os.sep + 'data' + os.sep + 'all_align_crop'

    transforms_train = []
    transforms_test = []

    transforms_train.append(transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.518,), std=(0.361,)),
                                                ]))

    transforms_test.append(transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.518,), std=(0.361,)),
                                               ]))
    transforms_train.append(A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        # A.OneOf([
        #     A.RandomContrast(),
        #     A.RandomGamma(),
        #     A.RandomBrightnessContrast(),
        # ], p=0.3),
        # A.OneOf([
        #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     A.GridDistortion(),
        #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        # ], p=0.3),
    ]))
    pairs = getPairs(data_path)
    train_dataset = InsertDataset(pairs, transforms_train)
    test_dataset = InsertDataset(pairs, transforms_test, is_train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

    return train_loader, test_loader


def train(args, model, optimizer, train_loader, device, writer, epoch, **kwargs):
    model.train()
    # if epoch < 1:
    #     model.require_encoder_grad(False)
    # else:
    #     model.require_encoder_grad(True)

    with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc='Epoch:{}'.format(epoch)) as t:
        for batch_idx, (data) in t:
            # if batch_idx == 0 or batch_idx == 20:
            #     print(model.check(model.first_conv))
            #     print(model.check(model.features2))
            data['im1'] = data['im1'].to(device)
            data['im2'] = data['im2'].to(device)
            data['mask'] = data['mask'].to(device)

            output = model(data)

            loss = calculate_loss(output, data['mask'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss * 0.9 + loss.item() * 0.1 if batch_idx != 0 else loss.item()

            t.set_postfix(loss=running_loss)

            if args.tensorboard and batch_idx % 10 == 0:
                writer.add_scalar('Train Loss / batch_idx', loss.item(), batch_idx + len(train_loader) * epoch)
                writer.add_scalar('Train Running Loss / batch_idx', running_loss, batch_idx + len(train_loader) * epoch)


def test(args, model, test_loader, device, writer, epoch, **kwargs):
    model.eval()
    total_loss = 0.
    total_iou_list = []

    miou_correct = [0] * 2
    miou_num = [0] * 2

    with torch.no_grad():
        for batch_idx, (data) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=100,
                                             desc='Epoch:{}'.format(epoch)):
            data['im1'] = data['im1'].to(device)
            data['im2'] = data['im2'].to(device)
            data['mask'] = data['mask'].to(device)

            output = model(data)

            total_loss += (data['im1'].shape[0] * calculate_loss(output, data['mask']).item())

            output = classification(output)

            correct_num, num = MIoU(output, data['mask'])
            miou_correct[0], miou_correct[1] = miou_correct[0] + correct_num[0], miou_correct[1] + correct_num[1]
            miou_num[0], miou_num[1] = miou_num[0] + num[0], miou_num[1] + num[1]

            iou_list = get_iou(data['mask'], output)
            total_iou_list = total_iou_list + iou_list

    total_loss /= len(test_loader.dataset)
    Avg_iou = 100. * np.mean(total_iou_list)  # 所有检测图片的平均iou
    miou_acc = 100. * (miou_correct[0] / miou_num[0] + miou_correct[1] / miou_num[1]) / 2
    miou_acc_class1 = 100. * miou_correct[1] / miou_num[1]
    print(
        '\nTest set: Average loss: {:.4f},  MIoU accuracy: {:.2f}%,  Class1 IoU accuracy: {:.2f}%'.format(
            total_loss, miou_acc, miou_acc_class1))
    print("\nAvg_iou: {:.5f}%".format(Avg_iou))


    if args.tensorboard:
        writer.add_scalar('Test Loss / epoch', total_loss, epoch)
        writer.add_scalar('Test MIoU accuracy / epoch', Avg_iou, epoch)
        writer.add_scalar('Test class1 IoU accuracy / epoch', miou_acc_class1, epoch)


def execute_one_epoch(epoch, args, model, scheduler, param, **kwargs):
    start_time = time.time()
    train(**param)
    waste_time1 = time.time() - start_time
    if args.is_k_fold:
        test(**param)
    waste_time2 = time.time() - start_time - waste_time1
    print('training wasting time:{:.0f}s, testing wasting time:{:.0f}s'.format(waste_time1, waste_time2))

    if epoch % args.save_model_interval == 0:
        if args.save:
            state = {'model': model.state_dict(), 'epoch': epoch}
            torch.save(state, args.checkpoint_path)

    scheduler.step()


def main():
    args = get_args()

    # 设置种子
    set_random_seed(args.seed)

    # 设置device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device(args.cuda if use_cuda else "cpu")

    # 定义模型和优化器
    model = U_Net_down3(in_channel=1, out_channel=1).to(device)
    # pretrained_model = torch.load(args.checkpoint_path, map_location=device)
    # model.load_state_dict(pretrained_model['model'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 45, 0.1)

    # 定义数据集
    if args.is_k_fold:
        train_loader, test_loader = data_load(args, k_index=1)

    else:
        train_loader, test_loader = data_load(args)

    # 定义tensorboard writer
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    param = {'args':args,
             'model':model,
             'optimizer':optimizer,
             'scheduler': scheduler,
             'train_loader':train_loader,
             'test_loader':test_loader,
             'device':device,
             'writer':writer,
             'epoch':None}

    for epoch in range(args.total_epochs):
        param['epoch'] = epoch + 1
        execute_one_epoch(**param, param=param)


if __name__ == '__main__':
    main()