import sys
from sys import path
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import glob
import cv2
import os
import random
import time

torch.manual_seed(42)

path0 = os.getcwd()
path1 = os.path.join(path0, sys.argv[1])
path.append(path1)
from Model import DetModel

def getPairs(base_dir, model_list=None):
    ng_files = []
    for d in glob.glob(base_dir + '/NG/*'):
        if not os.path.isdir(d):
            continue
        model_name = d.split('/')[-1]
        if model_list is not None and model_name not in model_list:
            continue

        for f in glob.glob(d + '/*'):
            if not os.path.isdir(f):
                continue
            for h in glob.glob(f + '/*.jpg'):
                h2 = h[:-4] + '_mask.bmp'
                ng_files.append((h, h2))

    pairs = []
    for d in ng_files:
        f = d[0].split('/')
        ok_dir = os.path.join(base_dir, 'OK', f[-3], f[-2])
        for f in glob.glob(ok_dir + '/*.jpg'):
            pairs.append((f, d[0], d[1]))
    random.shuffle(pairs)
    return pairs

def get_iou(query_mask,pred_label,mode='foreground'):
    if mode=='background':
        query_mask=1-query_mask
        pred_label=1-pred_label
    num_img=query_mask.shape[0]
    num_predict_list,inter_list,union_list,iou_list=[],[],[],[]
    for i in range(num_img):
        num_predict=torch.sum((pred_label[i]>0).float()).item()
        combination = (query_mask[i] + pred_label[i]).float()
        inter = torch.sum((combination == 2).float()).item()
        union = torch.sum((combination ==1).float()).item()+torch.sum((combination ==2).float()).item()
        if union!=0:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(inter/union)
            num_predict_list.append(num_predict)
        else:
            inter_list.append(inter)
            union_list.append(union)
            iou_list.append(0)
            num_predict_list.append(num_predict)
    return iou_list


def vis(outputs, batch_size_test, k):
    for j in range(outputs.size(0)):

        ##################################################################
        mask = outputs[j].squeeze(0).cpu().numpy() * 255
        mask = mask.astype(np.uint8)

        ######################################################################
        if not os.path.exists('save_images'):
            os.mkdir('save_images')
        cv2.imwrite('save_images/' + str(k * batch_size_test + j) + '.jpg', mask)    


class InsertDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, is_train=True):
        self.pairs = pairs
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        im1 = cv2.imread(self.pairs[idx][0], cv2.IMREAD_GRAYSCALE) #(512, 512)
        im2 = cv2.imread(self.pairs[idx][1], cv2.IMREAD_GRAYSCALE) #(512, 512)
        msk = cv2.imread(self.pairs[idx][2], cv2.IMREAD_GRAYSCALE) #msk
        msk = msk.astype(np.float32) / 255

        im1 = np.expand_dims(im1, axis=0) #(1, 512, 512)
        im2 = np.expand_dims(im2, axis=0) #(1, 512, 512)
        msk = np.expand_dims(msk, axis=0) #(1, 512, 512)

        return {'im1': im1, 'im2': im2}, {'mask': msk} #########

def test_model(test_pairs, model_path="", vis_flag=False):
      
    batch_size_test = 16
    ###############################################################################
    net = DetModel()
    net.load_state_dict(torch.load(model_path))
    print("load model successfully!")
    ###############################################################################################################
    
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()

    test_set = InsertDataset(test_pairs, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, num_workers=10)
    total_num = len(test_loader)
          
    total_iou_list = []
    total_time1 = time.time()
    net.eval()
    for k, data in enumerate(test_loader):
        if cuda:
            data_to_predict = data[0]
            data_to_predict['im1'] = data_to_predict['im1'].cuda().float() #([bs, 1, 512, 512])
            data_to_predict['im2'] = data_to_predict['im2'].cuda().float() #([bs, 1, 512, 512])

            data_mask = data[1]
            data_mask['mask'] = data_mask['mask'].cuda() #([bs, 1, 512, 512])

        time1 = time.time()
        outputs = net.predict(data_to_predict)
        time2 = time.time()
        predict_time_single = time2 - time1


        if vis_flag ==True:
            vis(outputs, batch_size_test, k)

        ################################ caculate iou ###############################################
        iou_list = get_iou(data_mask['mask'], outputs)
        iou_single = np.mean(iou_list)
        total_iou_list = total_iou_list + iou_list       
        if k % 100 == 0:
            print("im_predict: {:d}/{:d}, iou: {}, time: {}s".format(k + 1, total_num, iou_single, predict_time_single))
    
    total_time2 = time.time() 
    total_time = total_time2 - total_time1
    Avg_iou = np.mean(total_iou_list)
    print("Avg_iou: {}, total_time: {}s".format(Avg_iou, total_time))        


if __name__ == "__main__":

    model_path = os.path.join('.', "Model_parameter.p") 
    # with open('test_model_list.txt', 'r') as f:
    #     test_model_list = f.readlines()
    #     test_model_list = [f.strip() for f in test_model_list]

    test_pairs = getPairs(sys.argv[1], None)
    print('No. of testing pairs: {}'.format(len(test_pairs)))

    vis_flag = False
    test_model(test_pairs, model_path, vis_flag)

    
