# import lib
import argparse
import copy
import pickle
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
import PIL.Image as Image
import random
import math
from matplotlib import pyplot as plt  # plt提供画图工具

import torchvision.transforms.functional as F
import cv2
import ssl

from Reconstruct_minibatch import HCGLA_generator
from dataset import getDataset
from models import getModel, LeNetZhu, Gradinversion_lenet, ResNet18, Gradinversion_res18
from utils import compress_all

ssl._create_default_https_context = ssl._create_unverified_context


# 图片格式转换
def image_to_tensor(image, shape_img, device):
    transform1 = transforms.Compose([
        transforms.CenterCrop((shape_img)),  # 只能对PIL图片进行裁剪
        transforms.ToTensor(),
    ])
    dummy_data = transform1(image)
    dummy_data = torch.unsqueeze(dummy_data, 0)
    dummy_data = dummy_data.to(device).requires_grad_(True)
    return dummy_data


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = 'cuda:0'
else:
    device = 'cpu'
print(device)


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def grad2InitImg(attacked_model, gi_model, original_img, attacked_model_type="res18"):
    attacked_model.to(device)
    gi_model.to(device)
    original_img = original_img.view(-1, 3, 32, 32)
    pred = attacked_model(original_img.view(-1, 3, 32, 32))
    loss = torch.nn.CrossEntropyLoss()(pred, torch.LongTensor([0]).to(device))
    grad = torch.autograd.grad(loss, attacked_model.parameters(), retain_graph=True)
    if attacked_model_type == "res18":
        grad_input = grad[-2][:10].reshape(10, 32768, 1, 1)
    if attacked_model_type == "lenet":
        grad_input = grad[-2][:20].reshape(20, 768, 1, 1)
        # grad_input += torch.normal(mean=0.,std=0.1,size=grad_input.size()).to(device)
    # else:
    #    print("Undefined attacked_model_type")
    #    return
    recons = gi_model(grad_input)
    return recons


def HCGLA_generator(original_img, attacked_model_type="lenet"):
    if attacked_model_type == "lenet":
        lenet = LeNetZhu(num_classes=20, num_channels=3).to(device)
        gi_lenet = Gradinversion_lenet().to(device)
        gi_lenet.load_state_dict(torch.load("./models/generators/gi_lenet_epoch500_2000img(formal_class_20).pkl", map_location=torch.device(device)))
        recons = grad2InitImg(lenet, gi_lenet, original_img, attacked_model_type="lenet")
        return recons
    elif attacked_model_type == "res18":
        resnet18 = ResNet18(num_classes=10).to(device)
        gi_res18 = Gradinversion_res18().to(device)
        gi_res18.load_state_dict(torch.load("./models/generators/gi_res18_epoch1500_1500img(formal_class_10).pkl"))
        recons = grad2InitImg(resnet18, gi_res18, original_img, attacked_model_type="res18")
        return recons
    else:
        print("unknown model type")
        exit()


def recovery(opt, id):
    transform = transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        transforms.ToTensor()]) #
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    # shape_img = (32, 32)
    num_classes = 10
    channel = 3
    dst, num_classes, channel = getDataset(opt.dataset, opt.model_type)
    net = getModel(num_classes, channel, opt.model_type).to(device)
    if opt.is_exist:
        net.load_state_dict(torch.load(opt.model_path, map_location=device))

    criterion = nn.CrossEntropyLoss().to(device)
    # num_exp = 1
    ''' train DLG and iDLG 循环变量是图片的次序 '''
    # 索引保存下来 因为后面需要根据这个索引去获取到真实的图像

    '''
    # 找相同类别的图像
    print("find same label images")

    for ii in range(len(dst)):
        if ii != idx and dst[ii][1] == dst[idx][1]:
            same_labels_images.append(ii)
            print("find it!!")
            break
    '''
    gt_data = dst[id][0].to(device)
    if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
        gt_data = torch.cat([gt_data,gt_data,gt_data],dim=0)
        gt_name = opt.dataset + "_" + str(id) + ".jpg"
    else:
        gt_name = dst.imgs[id]
    gt_name = os.path.basename(gt_name)
    gt_data = gt_data.view(1, gt_data.shape[0], gt_data.shape[1], gt_data.shape[2])
    gt_label = torch.tensor([dst[id][1]]).to(torch.int64).to(device)
    gt_input = gt_data
    out = net(gt_input)
    if opt.model_type == "ResNet18" and opt.dataset == "CelebA":
        gt_label_input = int(gt_label) * 499 // 10176
        gt_label_input = torch.tensor(np.asarray(gt_label_input)).to(torch.int64).to(device).view(1)
    elif opt.model_type == "ResNet18" and opt.dataset == "lfw":
        gt_label_input = int(gt_label) * 499 // 5748
        gt_label_input = torch.tensor(np.asarray(gt_label_input)).to(torch.int64).to(device).view(1)
    elif opt.model_type == "ResNet18" and opt.dataset == "ImgNet":
        gt_label_input = int(gt_label) * 499 // 999
        gt_label_input = torch.tensor(np.asarray(gt_label_input)).to(torch.int64).to(device).view(1)
    else:
        gt_label_input = gt_label
    y = criterion(out, gt_label_input)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone().cpu() for _ in dy_dx))
    dy_dx, mask_tuple = compress_all(original_dy_dx, opt.compress_rate)
    original_dy_dx = [i.to(device) for i in dy_dx]

    # ,"prop_inf" , "img" , "Random" , "min_gloss"
    save_filename = '{}{}_{}'.format(opt.save_path, opt.model_type, opt.dataset)
    GT_path = '{}/{}'.format(save_filename, 'GT')
    Noisy_path = '{}/{}'.format(save_filename, 'Noisy')
    print("当前正在攻击图像序号:%d,标签是:%d" % (id, dst[id][1]))
    if not os.path.exists(save_filename):
        os.mkdir(save_filename)
    if not os.path.exists(GT_path):
        os.mkdir(GT_path)
    if not os.path.exists(Noisy_path):
        os.mkdir(Noisy_path)
    if opt.init_method == "Random":
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        # 产生 输出层大小的label,这的label也是一个向量，而不是具体的某个类
        print("生成随机噪点初始化")
    elif opt.init_method == "same_label":
        print("find same label images")
        same_labels_images = []
        for ii in range(len(dst)):
            # if ii != idx and int(dst[ii][1]) == int(dst[idx][1]):
            if dst[ii][1] == dst[id][1]:
                same_labels_images.append(ii)
                break
        idx1 = same_labels_images[0]
        dummy_data = dst[idx1][0].float().to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size())
        dummy_data = dummy_data.requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("相同标签初始化完成")
    elif opt.init_method == 'RGB':
        if opt.dataset == 'MNIST':
            dummy_image = Image.new('1', (32, 32), 'black')
        else:
            dummy_image = Image.new('RGB', (32, 32), 'red')
        dummy_data = image_to_tensor(dummy_image, (32, 32), device)
        # 产生 输出层大小的label,这的label也是一个向量，而不是具体的某个类
        init_data = copy.deepcopy(dummy_data)
        print("RGB初始化完成")
    elif opt.init_method == "ori_img":
        tmp_img = tp(dst[id][0])
        tmp_img = np.array(tmp_img)
        tmp_img.flags.writeable = True
        # 随机改变图像像素点
        c, w, h = tmp_img.shape
        total_pixels = c * w * h
        # @frac: 随机加噪的比例
        frac = 0.2
        num_random = int(frac * total_pixels)
        # 进行随机加噪
        for idx_1 in range(num_random):
            first = random.randint(0, c - 1)
            second = random.randint(0, w - 1)
            thrid = random.randint(0, h - 1)
            tmp_img[first][second][thrid] = 0

        # dummy_data = torch.from_numpy(tmp_img).float().to(device)
        tmp_tmp = Image.fromarray(tmp_img)
        dummy_data = tt(tmp_tmp).float().to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size())
        dummy_data = dummy_data.requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("原始图片初始化完成")
    elif opt.init_method == "img":
        # 特定图片初始化
        img = Image.open(opt.img_path)
        image_initial = transform(img)
        image_initial = image_initial.to(device)
        dummy_data = image_initial.view(1, image_initial.shape[0], image_initial.shape[1],
                                        image_initial.shape[2])  # .to(device).detach()
        dummy_data = dummy_data.requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("一张图片初始化完成")
    elif opt.init_method == "generator":
        if opt.model_type == "LeNet":
            dummy_data = HCGLA_generator(gt_data[0], attacked_model_type="lenet")
        elif opt.model_type == "ResNet18":
            dummy_data = HCGLA_generator(gt_data[0], attacked_model_type="res18")
        else:
            print("Undefined attacked_model_type")
            exit()
        dummy_data = tt(tp(dummy_data)).to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size()).requires_grad_(True)
        print("生成器初始化完成")
    else:
        print("Someting is wrong with the initial method, try it again!")
        return 0

    # 定义优化器
    if opt.method == 'DLG':
        dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=opt.lr)
    if opt.method == 'geiping':
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
        print("预测的标签是:", label_pred)
    else:
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
        print("预测的标签是:", label_pred)

    losses = []
    mses = []
    train_iters = []
    print('lr =', opt.lr)
    for iters in range(opt.Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            # dummy_dy_dx的非topk位置置0
            dummy_dy_dx = list(dummy_dy_dx)

            if opt.method == 'HCGLA':
                i = 0
                for tmp_1, tmp_2 in zip(dummy_dy_dx, mask_tuple):
                    dummy_dy_dx[i] = torch.mul(tmp_1, tmp_2.to(device))
                    i += 1
            grad_diff = 0
            if opt.method == 'geiping':
                # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                # if opt.model_type == "LeNet":
                ex = original_dy_dx[0]
                weights = torch.arange(len(original_dy_dx), 0, -1, dtype=ex.dtype, device=ex.device)/ len(original_dy_dx)
                for ii in range(len(original_dy_dx)):
                    grad_diff += 1 - torch.cosine_similarity(dummy_dy_dx[ii].flatten(), original_dy_dx[ii].flatten(), 0, 1e-10) * weights[ii]
                    # grad_diff += ((dummy_dy_dx[ii] - original_dy_dx[ii]) ** 2).sum()
                grad_diff += total_variation(dummy_data)
            elif opt.method == 'HCGLA':
                if opt.model_type == "LeNet":
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                else:
                    gx = dummy_dy_dx[-2]
                    gy = original_dy_dx[-2]
                    grad_diff += ((gx - gy) ** 2).sum()
            else:
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
            # print("grad_diff:", grad_diff)
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
    if mses[-1] < 0.1:
        gt_img = tp(gt_data[0].cpu())
        gt_img.save('%s/%s' % (GT_path, gt_name))
        dummy_img = tp(transform(tp(dummy_data[0].cpu())))
        dummy_name = '%s/%s' % (
            Noisy_path, gt_name)
        dummy_img.save(dummy_name)
        print("save")
    else:
        print("don't save")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_rate', default=0.001, type=float, help='The rate we compress our gradients')

    # about model
    parser.add_argument('--model_type', default='LeNet', help='you can choose from LeNet, ResNet18, ResNet34')
    parser.add_argument('--model_path', default='', help=" model path ")
    parser.add_argument('--is_exist', default=False, help='Is the attacked model exist ? or we need to random init it')

    # about data
    parser.add_argument('--dataset', default='CelebA',
                        help='you can choose from CelebA, lfw, pubface, google, cifar100, ImgNet')

    # init method
    parser.add_argument('--init_method', default='generator',
                        help="you can choose 'Random', 'same_label', 'ori_img', 'RGB', 'img', 'generator'")

    # save path
    parser.add_argument('--save_path', default='./data/noiseAndclearImage/', help='the path to save recover images')

    # recovery method
    parser.add_argument('--method', default='HCGLA', help='you can choose from DLG, iDLG, geiping, HCGLA')

    # hyper parameter
    parser.add_argument('--img_path', default='./data/img.jpg', help='')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    parser.add_argument('--Iteration', default=5, type=int, help='')
    parser.add_argument('--aux_path', default='./data/Auxiliary', help='')

    opt = parser.parse_args()
    sImages = random.sample(range(0, 60000), 20)
    for i in sImages:
        recovery(opt, i)
        print("\n------------------------%d/%d----------------------\n\n\n"%(i+1, len(sImages)))