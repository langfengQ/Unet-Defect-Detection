import numpy as np
import os
from PIL import Image
import glob
import h5py
import random


def gather_all_img_list(p):
    all_data = []
    for path in glob.glob(os.path.join(p, '*')):
        print(path)
        for sub_path in glob.glob(os.path.join(path, '*')):
            print(sub_path)
            for subsub_path in glob.glob(os.path.join(sub_path, '*')):
                print(subsub_path)
                all_data += glob.glob(os.path.join(subsub_path, '*.jpg'))
    return all_data


def data_label_processing(img_path, label_path):

    data_temp = np.array(Image.open(img_path))
    label_temp = np.array(Image.open(label_path))
    data = np.expand_dims(data_temp, 2)   # 添加通道维度
    label = np.zeros(label_temp.shape + (2,), label_temp.dtype)
    label[..., 0] = (label_temp > 200).astype(np.uint8)  # 缺陷部分
    label[..., 1] = (label_temp < 200).astype(np.uint8)  # ok部分

    return data, label


def data_label_list(path):
    # path = os.getcwd() + os.sep + "data" + os.sep + "all_align_crop"

    data_path_list = gather_all_img_list(path)
    label_path_list = []
    for img_path in data_path_list:
        label_path = img_path[:-4] + "_mask.bmp"
        label_path_list.append(label_path)
    # data, label = data_label_processing(img_path_list, label_path_list)
    return data_path_list, label_path_list


def getPairs(base_dir, model_list=None):
    ng_files = []
    for d in glob.glob(base_dir + '/NG/*'):
        d = d[:len(base_dir)+3] + '/' + d[len(base_dir)+4:]
        if not os.path.isdir(d):
            continue
        model_name = d.split('/')[-1]
        if model_list is not None and model_name not in model_list:
            continue

        for f in glob.glob(d + '/*'):
            f = f[:len(d)] + '/' + f[len(d)+1:]
            if not os.path.isdir(f):
                continue
            for h in glob.glob(f + '/*.jpg'):
                h = h[:len(f)] + '/' + h[len(f) + 1:]
                h2 = h[:-4] + '_mask.bmp'
                ng_files.append((h, h2))

    pairs = []
    for d in ng_files:
        f = d[0].split('/')
        ok_dir = os.path.join(base_dir, 'OK', f[-3], f[-2])
        # for f in glob.glob(ok_dir + '/*.jpg'):
            # pairs.append((f, d[0], d[1]))
        list_ = glob.glob(ok_dir + '/*.jpg')
        if len(list_) != 0:
            pairs.append((random.sample(list_, 1)[0], d[0], d[1]))
        else:
            pairs.append((d[0], d[0], d[1]))
    random.shuffle(pairs)
    return pairs


# def getPairs(base_dir, model_list=None):
#     ng_files = []
#     for d in glob.glob(base_dir + '/NG/*'):
#         d = d[:len(base_dir)+3] + '/' + d[len(base_dir)+4:]
#         if not os.path.isdir(d):
#             continue
#         model_name = d.split('/')[-1]
#         if model_list is not None and model_name not in model_list:
#             continue
#
#         for f in glob.glob(d + '/*'):
#             f = f[:len(d)] + '/' + f[len(d)+1:]
#             if not os.path.isdir(f):
#                 continue
#             for h in glob.glob(f + '/*.jpg'):
#                 h = h[:len(f)] + '/' + h[len(f) + 1:]
#                 h2 = h[:-4] + '_mask.bmp'
#                 ng_files.append((h, h2))
#
#     pairs = []
#     ok_pairs = []
#     list_pre = []
#     for d in ng_files:
#         f = d[0].split('/')
#         ok_dir = os.path.join(base_dir, 'OK', f[-3], f[-2])
#         # for f in glob.glob(ok_dir + '/*.jpg'):
#             # pairs.append((f, d[0], d[1]))
#         list_ = glob.glob(ok_dir + '/*.jpg')
#         if len(list_) != 0:
#             if list_ != list_pre:
#                 for ok in list_:
#                     msk_ok = ok[:-4] + '_mask.bmp'
#                     ok_pairs.append((ok, ok, msk_ok))
#             pairs.append((random.sample(list_, 1)[0], d[0], d[1]))
#             list_pre = list_
#         else:
#             pairs.append((d[0], d[0], d[1]))
#     random.shuffle(pairs)
#     random.shuffle(ok_pairs)
#     return (pairs, ok_pairs)

if __name__ == "__main__":
    # 处理OK数据
    print("Start processing OK img")
    path = os.getcwd() + os.sep + "data" + os.sep + "all_align_crop"
    getPairs(path)



