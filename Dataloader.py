from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
import glob
import argparse
import torch
import cv2
from utils import *


def calculate_mean_var(all_data_list):
    buffer = np.zeros((len(all_data_list), 512, 512), dtype=np.uint8)
    running_mean = 0
    running_var = 0

    for i in range(len(all_data_list)):
        buffer[i, ...] = np.array(Image.open(all_data_list[i]))
        # data = np.array(Image.open(all_data_list[i])) / 255
        # running_mean = 0.1 * data.mean() + 0.9 * running_mean
        # running_var = 0.1 * data.var() + 0.9 * running_var
        if i % 1000 == 0:
            print("{:.1f}%".format(100 * i / len(all_data_list)))
            # print(running_mean, ",,,,,", running_var)

    print(buffer.shape)

    mean = buffer.mean()
    var = buffer.std()

    return mean, var


def gather_all_img_list(p):
    all_data_list = []
    for path in glob.glob(os.path.join(p, '*')):
        # print(path)
        for sub_path in glob.glob(os.path.join(path, '*')):
            # print(sub_path)
            for subsub_path in glob.glob(os.path.join(sub_path, '*')):
                # print(subsub_path)
                all_data_list += glob.glob(os.path.join(subsub_path, '*.jpg'))

    return all_data_list


def data_label_processing(img_path, label_path):

    data = np.array(Image.open(img_path))
    mask = np.array(Image.open(label_path))

    path = os.getcwd() + os.sep + 'data' + os.sep + 'all_align_crop'
    label = 0 if label_path[len(path)+1: len(path)+3] == 'OK' else 1

    # label = np.expand_dims(label, 0)   # 添加通道维度
    # label[label > 0] = 1
    # label.astype(np.uint8)
    # label = np.zeros((2,) + label_temp.shape, label_temp.dtype)
    # label[0, ...] = (label_temp > 200).astype(np.uint8)  # 缺陷部分
    # label[1, ...] = (label_temp < 200).astype(np.uint8)  # ok部分

    # data = Image.fromarray(data)
    # label = Image.fromarray(label)

    return data, mask, label


def Clear_OK(path, train_data_path_list, train_label_path_list, test_data_path_list=None, test_label_path_list=None):
    train_data_path_list_re = []
    train_label_path_list_re = []
    test_data_path_list_re = []
    test_label_path_list_re = []
    for data, label in zip(train_data_path_list, train_label_path_list):
        if data[len(path) + 1: len(path) + 3] == 'NG':
            train_data_path_list_re.append(data)
            train_label_path_list_re.append(label)
    if test_data_path_list is not None:
        for data, label in zip(test_data_path_list, test_label_path_list):
            if label[len(path) + 1: len(path) + 3] == 'NG':
                test_data_path_list_re.append(data)
                test_label_path_list_re.append(label)

    return train_data_path_list_re, train_label_path_list_re, test_data_path_list_re, test_label_path_list_re


def data_label_list(path, is_k_fold=False, k=10, index=0, is_classification=True):
    data_path_list = gather_all_img_list(path)
    label_path_list = []
    for img_path in data_path_list:
        label_path = img_path[:-4] + "_mask.bmp"
        label_path_list.append(label_path)

    if is_k_fold:
        train_data_path_list = []
        train_label_path_list = []
        test_data_path_list = []
        test_label_path_list = []
        fold_size = len(data_path_list) // k
        for i in range(fold_size):
            if i == (fold_size-1):
                train_data_path_list += data_path_list[i * k: i * k + index]
                train_data_path_list += data_path_list[i * k + index + 1:]

                train_label_path_list += label_path_list[i * k: i * k + index]
                train_label_path_list += label_path_list[i * k + index + 1:]
            else:
                train_data_path_list += data_path_list[i * k: i * k + index]
                train_data_path_list += data_path_list[i * k + index + 1: (i + 1) * k]

                train_label_path_list += label_path_list[i * k: i * k + index]
                train_label_path_list += label_path_list[i * k + index + 1: (i + 1) * k]

            test_data_path_list.append(data_path_list[i * k + index])
            test_label_path_list.append(label_path_list[i * k + index])

        if not is_classification:
            train_data_path_list, train_label_path_list, test_data_path_list, test_label_path_list=\
                Clear_OK(path, train_data_path_list, train_label_path_list, test_data_path_list, test_label_path_list)
            f = open('test_list1.txt', 'w')
            for ii in range(len(test_data_path_list)):
                f.write(str(test_data_path_list[ii]) + '\n')
            f.close()
        else:
            f = open('test_list2.txt', 'w')
            for ii in range(len(test_data_path_list)):
                f.write(str(test_data_path_list[ii]) + '\n')
            f.close()
        return train_data_path_list, train_label_path_list, test_data_path_list, test_label_path_list
    else:
        if not is_classification:
            data_path_list, label_path_list, _, _ = Clear_OK(path, data_path_list, label_path_list)
        return data_path_list, label_path_list, None, None


class My_Dataloader(Dataset):
    def __init__(self, data_list, is_train, transform=None, is_k_fold=True):
        self.data_list, self.label_list = data_list[0], data_list[1]
        if not is_train and is_k_fold:
            self.data_list, self.label_list = data_list[2], data_list[3]
        self.transform = transform
        self.num = len(self.data_list)

    def __getitem__(self, item):
        data, mask, label = data_label_processing(self.data_list[item], self.label_list[item])
        if len(self.transform) == 2:
            transformed = self.transform[1](image=data, mask=mask)
            data = transformed["image"]
            mask = transformed["mask"]
        data = self.transform[0](data)
        mask = transforms.ToTensor()(mask)
        return data, mask, label

    def __len__(self):
        return self.num


class InsertDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, transform, is_k=True, is_train=True):
        self.transform = transform
        if is_k:
            self.data = pairs[:int(len(pairs) / 10)] if not is_train else pairs

        if is_train:
            self.remove()

        self.num = len(self.data)

    def __len__(self):
        return self.num

    def remove(self):
        pairs = []
        remove_pairs = []
        for p in self.data:
            msk = p[2]
            img_msk = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
            rate = np.sum(img_msk > 200)
            if rate > 30:
                pairs.append(p)
            else:
                remove_pairs.append(p)

        self.data = pairs

    def __getitem__(self, idx):
        im1 = cv2.imread(self.data[idx][0], cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(self.data[idx][1], cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(self.data[idx][2], cv2.IMREAD_GRAYSCALE)

        if len(self.transform) == 2:
            im = np.zeros(im1.shape + (2,), dtype=np.uint8)
            im[..., 0] = im1
            im[..., 1] = im2
            transformed = self.transform[1](image=im, mask=msk)
            im = transformed["image"]
            msk = transformed["mask"]
            im1, im2 = im[..., 0], im[..., 1]
        im1 = self.transform[0](im1)
        im2 = self.transform[0](im2)
        msk = transforms.ToTensor()(msk)

        return {'im1': im1, 'im2': im2, 'mask': msk}


def get_mean_std():
    data_path = os.getcwd() + os.sep + 'data' + os.sep + 'all_align_crop'

    data_list, _, _, _ = data_label_list(data_path, is_k_fold=False, is_classification=False)

    data = []
    mean = []
    std = []
    for i, img_path in enumerate(data_list):
        if i % 1845 == 0 and i != 0:
            data_ = np.array(data)
            mean.append(data_.mean())
            std.append(data_.std())
            print('>>>>>>>>>>>>>>>>>>>{}/{}, {:.2f}%'.format(i, len(data_list), 100. * i / len(data_list)))
            print(mean[-1])
            print(std[-1])
            del data[:]
        data.append(np.array(Image.open(img_path)) / 255.)

    data = np.array(data)
    mean.append(data.mean())
    std.append(data.std())
    print('>>>>>>>>>>>>>>>>>>>')
    print(mean[-1])
    print(std[-1])
    print('<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>')
    print(np.array(mean).mean())
    print(np.array(std).mean())


if __name__ == "__main__":
    get_mean_std()
