import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from skimage import morphology
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.5, device='cuda:0'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.device = device

    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        pred = torch.cat((1 - pred, pred), dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).to(self.device)
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.00001, max=1.0)
        log_p = probs.log()
        alpha = torch.ones(pred.shape[0], pred.shape[1]).to(self.device)
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        loss = batch_loss.mean()

        return loss


def dice_loss(prediction, label):
    """
    Calculating the dice loss
    """
    smooth = 1.0

    prediction = prediction.view(-1)
    label = label.view(-1)

    intersection = (prediction * label).sum()

    return 1 - ((2. * intersection + smooth) / (prediction.sum() + label.sum() + smooth))


def calculate_loss(prediction, label):
    """
    Calculating the loss and metrics
    """
    bce = F.binary_cross_entropy_with_logits(prediction, label)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, label)
    # focalLoss = FocalLoss(device=prediction.device)(prediction, label)

    loss = bce * 0.8 + dice * 0.2
    # loss = 5 * focalLoss + bce

    return loss


def calcuate_acc(prediction, label, threshold=0.1):
    """
    Calculating average acc
    """
    prediction = torch.sigmoid(prediction)
    prediction[prediction > threshold] = 1
    prediction[prediction <= threshold] = 0

    correct_num = prediction.numel() - torch.sum(abs(label - prediction))

    return correct_num.item(), prediction.numel()


def MIoU(prediction, label):
    """
    Calculating MIoU acc
    """

    num_l0, num_l1 = torch.sum(abs(label - 0) < 1e-5), torch.sum(abs(label - 1) < 1e-5)
    num_p0, num_p1 = torch.sum(abs(prediction - 0) < 1e-5), torch.sum(abs(prediction - 1) < 1e-5)

    num_p0l0 = torch.sum(abs(label[abs(prediction - 0) < 1e-5] - 0) < 1e-5, dtype=torch.float)
    num_p1l1 = torch.sum(abs(label[abs(prediction - 1) < 1e-5] - 1) < 1e-5, dtype=torch.float)

    return [num_p0l0.item(), num_p1l1.item()], \
           [(num_p0 + num_l0 - num_p0l0).item(), (num_p1 + num_l1 - num_p1l1).item()]


def classification(prediction, threshold=0.1):
    device = prediction.device
    prediction = torch.sigmoid(prediction)
    prediction[prediction > threshold] = 1
    prediction[prediction <= threshold] = 0

    # prediction = area_connection(prediction)
    prediction = prediction.to(device)

    # w = 0.599
    # result = prediction * w + pred * ((1 - w) / threshold)
    #
    # result[result >= 1] = 1

    return prediction


def area_connection(prediction, area_threshold=20):
    # 去除小物体t
    prediction = prediction.cpu().numpy().astype(np.uint8)
    prediction = morphology.remove_small_objects(prediction == 1, min_size=area_threshold,
                                                 connectivity=1, in_place=True)

    # 获取最终label
    prediction = torch.Tensor(prediction)

    return prediction


def show(data, label):
    # output = torch.sigmoid(output)
    # output[output > 0.5] = 1
    # output[output <= 0.5] = 0

    data0 = data[0]
    data1 = data[1]
    label0 = label[0]
    label1 = label[1]

    plt.matshow(data0)
    plt.matshow(data1[0, ...])
    plt.matshow(label0)
    plt.matshow(label1)
    plt.show()

    return None


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 2, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

