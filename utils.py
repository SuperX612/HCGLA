import copy
import math
import random

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data.sampler import Sampler
from collections import defaultdict


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


def compress_one(gradient, compress_rate):
    c = gradient.flatten().abs().sort()[0]
    threshold = c[-int(len(c) * compress_rate)]
    temp = torch.ones(gradient.size())
    temp[gradient.abs() < threshold] = 0
    gradient[gradient.abs() < threshold] = 0
    return gradient, temp


def compress_onebyone(gradient, compress_rate):
    gradient = list((_.detach().clone() for _ in gradient))
    mask = []
    for i in range(len(gradient)):
        gradient[i], temp = compress_one(gradient[i], compress_rate)
        mask.append(temp)
    return gradient, mask


def compress_all(gradients, compress_rate):
    gradients = list((_.detach().cpu().clone() for _ in gradients))
    mask_tuple = []
    c = np.asarray(gradients[0])
    c = abs(c.ravel())
    mask_tuple.append(np.ones(gradients[0].shape))
    for x in gradients[1:]:
        a = np.asarray(x)  # 转化为array
        a = abs(a.ravel())
        c = np.append(c, a)
        mask_tuple.append(np.ones(x.shape))
    sort_c = np.sort(c)
    top = len(sort_c)
    standard = sort_c[int(-top * compress_rate)]
    print('compress shield : ', standard)
    newgra = copy.deepcopy(gradients)
    for i in range(len(newgra)):
        p = np.asarray(newgra[i])
        m = mask_tuple[i]
        m[abs(p) < standard] = 0
        p[abs(p) < standard] = 0
        mask_tuple[i] = torch.tensor(m)
        newgra[i] = torch.tensor(p)
    return newgra, mask_tuple


# def compress_all(gradient, compress_rate):
#     gradient = list((_.detach().clone() for _ in gradient))
#     mask = []
#     if compress_rate < 1.0:
#         c = torch.cat([gradient[i].flatten() for i in range(len(gradient))]).flatten().abs().sort()[0]
#         threshold = c[-int(len(c) * compress_rate)]
#         for i in range(len(gradient)):
#             temp = torch.ones(gradient[i].size())
#             temp[gradient[i].abs() < threshold] = 0
#             mask.append(temp)
#             gradient[i][gradient[i].abs() < threshold] = 0
#     else:
#         for i in range(len(gradient)):
#             temp = torch.ones(gradient[i].size())
#             mask.append(temp)
#     return gradient, mask


def getClassIdx(dataset):
    dic_class = defaultdict(list)
    idx = dataset.labs
    if not isinstance(idx, torch.Tensor):
        idx = torch.Tensor(dataset.labs)
    for i in range(dataset.labs.max()):
        dic_class[i] = torch.where(idx == i)[0].tolist()
    return dic_class


class BatchSampler(Sampler):
    # 批次采样
    def __init__(self, class_idx, batch_size, datasetnum, drop_last):
        # ...
        self.class_idx = class_idx
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.datasetnum = datasetnum

    def __iter__(self):
        batch = []
        for i in range(self.datasetnum // self.batch_size):
            classes = random.sample(range(0, len(self.class_idx)), self.batch_size)
            # classes = list(np.sort(np.asarray(classes)))
            for classn in classes:
                idx = random.choice(self.class_idx[classn])
                batch.append(idx)
            yield batch
            batch = []
        # 如果不需drop最后一组返回最后一组
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.datasetnum // self.batch_size
        else:
            return (self.datasetnum + self.batch_size - 1) // self.batch_size
