# show train result
import os
import random
import copy
import numpy as np
import torch
import torchvision.utils as vutils
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from dataset import Dataset_from_Image, CelebA_dataset
from models import LeNetZhu, Generator, Gradinversion_lenet

from utils import compress_one, getClassIdx, BatchSampler, compress_all, compress_onebyone


def imshow(images, nrow):
    img_batch = make_grid(images, nrow=nrow, padding = 0)
    img = img_batch
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # 将【3，32，128】-->【32,128,3】
    plt.axis('off')
    plt.show()


def show(modelpath, seed=100):
    class_num = 10177
    model_path = modelpath
    generator = Generator().to(device)
    att_model = LeNetZhu(num_classes=class_num, num_channels=3)
    att_model = att_model.to(device)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    train_dataset = CelebA_dataset()
    # train_dataset = lfw_dataset()
    class_idx = getClassIdx(train_dataset)
    batchSampler = BatchSampler(class_idx, batchsize, len(train_dataset), drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batchSampler)
    generator.eval()
    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)
    pred = att_model(x)
    loss = nn.CrossEntropyLoss()(pred, y.long())
    grad = torch.autograd.grad(loss, att_model.parameters(), retain_graph=True)
    # original_dy_dx = list((_.detach().clone().cpu() for _ in grad))
    # dy_dx, _  = compress_onebyone(grad, compress_rate)
    # dy_dx = dy_dx[-2]
    dy_dx = compress_one(grad[-2], compress_rate)[0]
    fullconnectGrad = dy_dx.reshape(class_num, 768, 1, 1).to(device)
    grad_input = fullconnectGrad[[int(t) for t in y], :]
    recons = generator(grad_input)
    imshow(torch.cat((x, recons), dim=0), batchsize)


if __name__ == "__main__":
    # model_path = "./models/generators/gi_lenet_epoch500_2000img(formal_class_20).pkl"
    model_path = "./models/generators/generator_batchsize4.pkl"
    # model_path = "../models/batch16Generator/generator_lr0.001_epoch56_globalLoss6377.71609.pkl"
    # model_path = "/home/b1107/user/xkl/HLA/models/com0.001batchsize16/generator_lr0.01_epoch50_globalLoss6645.24755.pkl"
    batchsize = 4
    compress_rate = 0.001
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    show(modelpath=model_path, seed=20)