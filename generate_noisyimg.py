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

from grad2img import HLA_generator
from gather_inference import attr_dict
from sklearn.preprocessing import Normalizer, StandardScaler
from modellib import LeNetZhu, ResNet18, ResNet34, DnCNN
import torchvision.transforms.functional as F
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


AUC_top1_1000 = [0.80, 0.80, 0.79, 0.73, 0.92, 0.88, 0.64, 0.71, 0.87,
                 0.93, 0.69, 0.81, 0.75, 0.81, 0.80, 0.85, 0.82, 0.91,
                 0.88, 0.83, 0.88, 0.78, 0.81, 0.62, 0.81, 0.64, 0.93,
                 0.69, 0.81, 0.85, 0.82, 0.82, 0.67, 0.82, 0.76, 0.92,
                 0.90, 0.74, 0.86, 0.75]
AUC_top10_100 = []
AUC_Full = []
AUC = AUC_top1_1000
# sImages = random.sample(range(0,13233), 1)  # The images we want to recover 10002, 54354, 25, 22
# 0.001: 10001, 898,

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, lab


def lfw_dataset():
    images_all = []
    labels_all = []
    lfw_path = '../data/lfw'
    folders = os.listdir(lfw_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def lfw_some_dataset(num=500):
    images_all = []
    labels_all = []
    lfw_path = '../data/lfw'
    folders = os.listdir(lfw_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold).replace('\\', '/'))
        if foldidx < num:
            for f in files:
                if len(f) > 4 and f[-4:] == '.jpg':
                    images_all.append(os.path.join(lfw_path, fold, f))
                    labels_all.append(foldidx)
        else:
            break

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def ImgNet_dataset():
    images_all = []
    labels_all = []
    ImgNet_path = "D://XiangKunlan//mean_teacher_wang//mean-teacher-master//pytorch//data-local//images//ilsvrc2012//train"
    folders = os.listdir(ImgNet_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(ImgNet_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-5:] == '.JPEG':
                images_all.append(os.path.join(ImgNet_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def CelebA_dataset():
    images_all = []
    labels_all = []
    # print("folders:", folders)
    CelebA_path = "../data/CelebA/Img/img_align_celeba"
    ATTR_DIR = '../data/CelebA/Anno/identity_CelebA.txt'

    with open(ATTR_DIR, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(CelebA_path, filename)
            images_all.append(filepath_old)
            labels_all.append(int(info[1]) - 1)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all), transform=transform)
    return dst


def CelebA_some_dataset(num=500):
    images_all = []
    labels_all = []
    # print("folders:", folders)
    CelebA_path = "../data/CelebA/Img/img_align_celeba"
    ATTR_DIR = '../data/CelebA/Anno/identity_CelebA.txt'

    with open(ATTR_DIR, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            if int(info[1]) <= num:
                filename = info[0]
                filepath_old = os.path.join(CelebA_path, filename)
                images_all.append(filepath_old)
                labels_all.append(int(info[1]) - 1)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all), transform=transform)
    return dst


def pubface_dataset():
    images_all = []
    labels_all = []
    pubface_path = '../data/pubface/train'
    folders = os.listdir(pubface_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(pubface_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(pubface_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def google_dataset():
    file = open("../data/google/list_attr.txt", 'r')
    google_path = '../data/google/images'
    folders = os.listdir(google_path)
    images_all = []
    labels_all = []
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(google_path, str(foldidx)).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(google_path, str(foldidx), f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


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


def compress(gradients, compress_rate):
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


def collect_grads(aggr_g):
    fcg = np.asarray(aggr_g[-2])
    local = np.argmax(abs(fcg), 0)
    ff = []
    for i in range(len(local)):
        ff.append(fcg[local[i], i])
    fcg = np.asarray(ff)
    return fcg

def ResNet18_prop_inf(gt_data, gt_label, compress_rate, forest_path):
    net = LeNetZhu(num_classes=10177, num_channels=3).to(device)
    out = net(gt_data)
    criterion = nn.CrossEntropyLoss()
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone().cpu() for _ in dy_dx))
    dy_dx, mask_tuple = compress(original_dy_dx, compress_rate)
    grad = collect_grads(dy_dx)
    pred = []
    best_img_path = '../data/CelebA/Img/img_align_celeba/000001.jpg'
    # property_path = '../data/CelebA/Anno/list_attr_celeba.txt'
    property_path = "../data/CelebA/Anno/list_attr_celeba.txt"
    data_path = "../data/CelebA/Img/img_align_celeba"
    maxscore = 0
    for i in range(40):
        filename = forest_path + '{}_AUC{:.2f}.pkl'.format(attr_dict[i + 1], AUC[i])
        with open(filename, 'rb') as f:
            clf = pickle.load(f)
            x_test = [grad]
            normalizer = Normalizer(norm='l2')
            X_test = normalizer.transform(x_test)
            # y_score = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)
            if int(y_pred) == 1:
                pred.append(1)
            else:
                pred.append(-1)
    print(pred)
    with open(property_path, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(data_path, filename)
            current_score = 0
            # if index == id:
            #     print(line)
            for i in range(40):
                if int(info[i + 1]) == pred[i]:
                    current_score += (AUC[i] - 0.5)
                else:
                    current_score -= (AUC[i] - 0.5)
            if current_score > maxscore:
                maxscore = current_score
                best_img_path = filepath_old
            index += 1
    return best_img_path


def prop_inf(grad, id, forest_path):
    grad = collect_grads(grad)
    pred = []
    best_img_path = '../data/CelebA/Img/img_align_celeba/000001.jpg'
    # property_path = '../data/CelebA/Anno/list_attr_celeba.txt'
    property_path = "../data/CelebA/Anno/list_attr_celeba.txt"
    data_path = "../data/CelebA/Img/img_align_celeba"
    maxscore = 0
    for i in range(40):
        filename = forest_path + '{}_AUC{:.2f}.pkl'.format(attr_dict[i + 1], AUC[i])
        with open(filename, 'rb') as f:
            clf = pickle.load(f)
            x_test = [grad]
            normalizer = Normalizer(norm='l2')
            X_test = normalizer.transform(x_test)
            # y_score = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)
            if int(y_pred) == 1:
                pred.append(1)
            else:
                pred.append(-1)
    print(pred)
    with open(property_path, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(data_path, filename)
            current_score = 0
            # if index == id:
            #     print(line)
            for i in range(40):
                if int(info[i + 1]) == pred[i]:
                    current_score += (AUC[i] - 0.5)
                else:
                    current_score -= (AUC[i] - 0.5)
            if current_score > maxscore:
                maxscore = current_score
                best_img_path = filepath_old
            index += 1
    return best_img_path


def getmin_gloss(auxdst, net, criterion, mask_tuple, original_dy_dx, label_pred, transform):
    x = transform(auxdst[0][0]).to(device)
    x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
    best_img = x
    min_gloss = 1000000
    for i in range(len(auxdst)):
        x = transform(auxdst[i][0]).to(device)
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        y_hat = net(x)
        y_loss = criterion(y_hat, label_pred)
        current_dy_dx = torch.autograd.grad(y_loss, net.parameters(), create_graph=True)
        current_dy_dx = list(current_dy_dx)
        j = 0
        for tmp_1, tmp_2 in zip(current_dy_dx, mask_tuple):
            current_dy_dx[j] = torch.mul(tmp_1, tmp_2.to(device))
            j += 1
        grad_diff = 0
        for gx, gy in zip(current_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        if min_gloss>grad_diff:
            min_gloss = grad_diff
            best_img = x
    return best_img


def total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def recovery(opt, id):
    transform = transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        transforms.ToTensor()]) #
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    # shape_img = (32, 32)
    num_classes = 10
    channel = 3
    ''' load data '''
    if opt.dataset == 'CelebA':
        if opt.model_type == "LeNet":
            dst = CelebA_dataset()
            num_classes = 10177
            channel = 3
        else:
            dst = CelebA_dataset()# CelebA_some_dataset(num=500)
            num_classes = 500
            channel = 3
    elif opt.dataset == 'lfw':
        if opt.model_type == "LeNet":
            dst = lfw_dataset()
            num_classes = 10177
            channel = 3
        else:
            dst = lfw_dataset()
            num_classes = 500
            channel = 3
    elif opt.dataset == 'pubface':
        dst = pubface_dataset()
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif opt.dataset == 'google':
        dst = google_dataset()
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif opt.dataset == 'cifar10':  # classes:10, counts:50000
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
        dst = datasets.CIFAR10("../data/cifar10", download=True, transform=transform)
    elif opt.dataset == 'cifar100':  # classes:100, counts:50000
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
        dst = datasets.CIFAR100("../data/cifar100", download=True, transform=transform)
    elif opt.dataset == "ImgNet":
        dst = ImgNet_dataset()
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif opt.dataset == "MNIST": # classes:10, counts:60000
        dst = datasets.MNIST("../data/MNIST", download=True, transform=transform)
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif opt.dataset == "FMNIST": # classes:10, counts:60000
        dst = datasets.MNIST("../data/Fashion_MNIST", download=True, transform=transform)
        if opt.model_type == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    else:
        exit('unknown dataset')

    if opt.model_type == 'LeNet':
        net = LeNetZhu(num_classes=num_classes, num_channels=channel).to(device)
    elif opt.model_type == 'ResNet18':
        net = ResNet18(num_classes=num_classes).to(device)
    elif opt.model_type == 'ResNet34':
        net = ResNet34(num_classes=num_classes).to(device)
    else:
        print("unknown model-type")
        exit()
    if opt.is_exist:
        net.load_state_dict(torch.load(opt.model_path, map_location='cpu'))

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
    dy_dx, mask_tuple = compress(original_dy_dx, opt.compress_rate)
    original_dy_dx = [i.to(device) for i in dy_dx]

    # ,"prop_inf" , "img" , "Random" , "min_gloss"
    save_filename = '{}{}_{}'.format(opt.save_path, opt.model_type, opt.dataset)
    GT_path = '{}/{}'.format(save_filename, 'GT')
    Noisy_path = '{}/{}'.format(save_filename, 'Noisy')
    print('%s - %s - Filter:%s' % (opt.method, opt.init_method, str(opt.filter)))
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
    elif opt.init_method == 'prop_inf':
        if opt.model_type == "LeNet":
            img_path = prop_inf(dy_dx, id, opt.forest_path)
        else:
            img_path = ResNet18_prop_inf(gt_data, gt_label, opt.compress_rate, opt.forest_path)
        print(img_path)
        dummy_data = Image.open(img_path)
        dummy_data = transform(dummy_data).to(device)
        dummy_data = dummy_data.view(1, *dummy_data.size()).requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("属性推断初始化完成")
    elif opt.init_method == "generator":
        if opt.model_type == "LeNet":
            dummy_data = HLA_generator(gt_data[0], attacked_model_type="lenet")
        elif opt.model_type == "ResNet18":
            dummy_data = HLA_generator(gt_data[0], attacked_model_type="res18")
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

            if opt.method == 'HLA':
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

            elif opt.method == 'HLA':
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


def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_rate', default=0.001, type=float, help='The rate we compress our gradients')

    # about model
    parser.add_argument('--model_type', default='LeNet', help='you can choose from LeNet, ResNet18, ResNet34')
    parser.add_argument('--model_path', default='', help=" model path ")
    parser.add_argument('--is_exist', default=False, help='Is the attacked model exist ? or we need to random init it')

    # about data
    parser.add_argument('--dataset', default='CelebA', help='you can choose from CelebA, lfw, pubface, google, cifar100, ImgNet')

    # init method
    parser.add_argument('--init_method', default='Random',
                        help="you can choose 'Random', 'same_label', 'ori_img', 'RGB', 'img', 'prop_inf', 'generator'")

    # save path
    parser.add_argument('--save_path', default='../GT_Noisy/others/', help='the path to save recover images')

    # recovery method
    parser.add_argument('--method', default='HLA', help='you can choose from DLG, iDLG, geiping, HLA')

    # hyper parameter
    parser.add_argument('--img_path', default='../data/img.jpg', help='')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    parser.add_argument('--Iteration', default=5, type=int, help='')
    parser.add_argument('--aux_path', default='../data/Auxiliary', help='')

    # about filter
    parser.add_argument('--filter', default=False, type=bool, help='')
    parser.add_argument('--filter_method', default='HLA-Filter',
                        help="you can choose from 'HLA-Filter','mean','Guassian','median','bilater' ")
    parser.add_argument('--filter_path', default='../models/filterModel/DnCNN/net_799.pth', help='')
    parser.add_argument('--filter_frequency', default=5, type=int)
    parser.add_argument('--isShow', default=True, help="")

    # perperty inference
    parser.add_argument('--forest_path', default='../models/randomForest/LeNet_batchsize1_comRate0.001/')

    opt = parser.parse_args()
    return opt


def one(opt, sImages):
    for i in sImages:
        recovery(opt, i)
        print("\n------------------------%d/%d----------------------\n\n\n"%(i+1, len(sImages)))


def main():
    opt = getOpt()
    # CelebA:202599, lfw:13233, pubface:9408, google:3570, cifar10:50000, cifar100:50000, ImgNet:1281167, MNIST:60000, FMNIST:60000
    sImages = random.sample(range(0, 60000), 20)
    opt.method = 'HLA' # DLG, iDLG, geiping, HLA
    opt.model_type = "ResNet18" # LeNet, ResNet18
    opt.init_method = 'generator' # Random, same_label, ori_img, RGB, img, prop_inf, generator
    opt.dataset = 'MNIST' # CelebA, lfw, pubface, google, cifar10, cifar100, ImgNet, MNIST, FMNIST
    opt.Iteration = 5
    opt.filter = False # True, False
    opt.filter_method = "HLA-Filter" # HLA-Filter, mean, Guassian, median, bilater
    opt.compress_rate = 0.001
    opt.isShow = False # True, False
    opt.filter_frequency = 5
    one(opt, sImages)


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
    parser.add_argument('--init_method', default='Random',
                        help="you can choose 'Random', 'same_label', 'ori_img', 'RGB', 'img', 'prop_inf', 'generator'")

    # save path
    parser.add_argument('--save_path', default='./data/noiseAndclearImage/', help='the path to save recover images')

    # recovery method
    parser.add_argument('--method', default='HLA', help='you can choose from DLG, iDLG, geiping, HLA')

    # hyper parameter
    parser.add_argument('--img_path', default='../data/img.jpg', help='')
    parser.add_argument('--lr', default=0.1, type=float, help='')
    parser.add_argument('--Iteration', default=5, type=int, help='')
    parser.add_argument('--aux_path', default='../data/Auxiliary', help='')

    opt = parser.parse_args()
    sImages = random.sample(range(0, 60000), 20)
    main()