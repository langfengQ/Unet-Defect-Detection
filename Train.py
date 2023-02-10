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

from model import U_Net
from Dataloader import My_Dataloader, data_label_list
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='trian')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size for training')
    parser.add_argument('--batch-size-test', type=int, default=16, help='batch size for testing')
    parser.add_argument('--total-epochs', type=int, default=50, help='number of epochs to train')
    # parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--use-cuda', action='store_false', help='use CUDA')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='CUDA number')
    parser.add_argument('--seed', type=int, default=9924, help='random seed')
    parser.add_argument('--is-k-fold', action='store_false', help='')
    parser.add_argument('--k', type=int, default=10, help='k of k_fold method')

    parser.add_argument('--save', action='store_true', help='save model')
    parser.add_argument('--tensorboard', action='store_true', help='write tensorboard')
    parser.add_argument('--tensorboard-path', type=str, default='./summaries/unet_onlytraindata_05/',
                        help='path of tensorboard')
    parser.add_argument('--save-model-interval', type=int, default=5,
                        help='save model every save_model_interval')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint/unet_model_onlytraindata_05.pth',
                        help='path of saved model')
    # parser.add_argument('--log-interval', type=int, default=20,
    #                     help='how many batches to wait before logging training status')

    args = parser.parse_args()
    return args


# 设置种子
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_load(args, k_index=1):
    data_path = os.getcwd() + os.sep + 'data' + os.sep + 'all_align_crop'

    transforms_train = []
    transforms_test = []

    transforms_train.append(transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.518,), std=(0.361,))]))

    transforms_test.append(transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.518,), std=(0.361,))]))
    transforms_train.append(A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        # A.OneOf([
        #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     A.GridDistortion(),
        #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        # ], p=1),
    ]))

    if args.is_k_fold:
        data_list = data_label_list(data_path, is_k_fold=args.is_k_fold, k=args.k, index=k_index, is_classification=False)
        train_dataset = My_Dataloader(data_list, is_train=True, transform=transforms_train,
                                         is_k_fold=args.is_k_fold)
        test_dataset = My_Dataloader(data_list, is_train=False, transform=transforms_test,
                                     is_k_fold=args.is_k_fold)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

    else:
        data_list = data_label_list(data_path, is_k_fold=args.is_k_fold, is_classification=False)
        train_dataset = My_Dataloader(data_list, is_train=True, transform=transforms_train,
                                         is_k_fold=args.is_k_fold)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = None

    return train_loader, test_loader


def train(args, model, optimizer, train_loader, device, writer, epoch, **kwargs):
    model.train()

    with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc='Epoch:{}'.format(epoch)) as t:
        for batch_idx, (data, mask, _) in t:
            data, mask = data.to(device), mask.to(device)
            output = model(data)

            loss = calculate_loss(output, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss * 0.95 + loss.item() * 0.05 if batch_idx != 0 else loss.item()

            t.set_postfix(loss=running_loss)

            if args.tensorboard and batch_idx % 10 == 0:
                writer.add_scalar('Train Loss / batch_idx', loss.item(), batch_idx + len(train_loader) * epoch)
                writer.add_scalar('Train Running Loss / batch_idx', running_loss, batch_idx + len(train_loader) * epoch)


def test(args, model, test_loader, device, writer, epoch, **kwargs):
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_num = 0

    miou_correct = [0] * 2
    miou_num = [0] * 2

    with torch.no_grad():
        for batch_idx, (data, mask, _) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=100,
                                             desc='Epoch:{}'.format(epoch)):
            data, mask = data.to(device), mask.to(device)
            output = model(data)

            total_loss += (data.shape[0] * calculate_loss(output, mask).item())

            correct_num, num = calcuate_acc(output, mask)
            total_correct += correct_num
            total_num += num

            correct_num, num = MIoU(output, mask)
            miou_correct[0], miou_correct[1] = miou_correct[0] + correct_num[0], miou_correct[1] + correct_num[1]
            miou_num[0], miou_num[1] = miou_num[0] + num[0], miou_num[1] + num[1]

    total_loss /= len(test_loader.dataset)
    total_acc = 100. * total_correct / total_num
    miou_acc = 100. * (miou_correct[0] / miou_num[0] + miou_correct[1] / miou_num[1]) / 2
    miou_acc_class1 = 100. * miou_correct[1] / miou_num[1]
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {:.5f}%,  MIoU accuracy: {:.2f}%,  Class1 MIoU accuracy: {:.2f}%'.format(
            total_loss, total_acc, miou_acc, miou_acc_class1))

    if args.tensorboard:
        writer.add_scalar('Test Loss / epoch', total_loss, epoch)
        writer.add_scalar('Test Accuracy / epoch', total_acc, epoch)
        writer.add_scalar('Test MIoU accuracy / epoch', miou_acc, epoch)


def execute_one_epoch(epoch, args, model, scheduler, param, **kwargs):
    start_time = time.time()
    # train(**param)
    if args.is_k_fold:
        test(**param)
    waste_time = time.time() - start_time
    print('One epoch wasting time:{:.0f}s'.format(waste_time))

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
    model = U_Net(in_channel=1, out_channel=1).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

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
        param['epoch'] = epoch
        execute_one_epoch(**param, param=param)


if __name__ == '__main__':
    main()