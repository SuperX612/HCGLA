# import lib
import argparse
import copy
import pickle
import time
import os
from collections import defaultdict
from torch.utils.data.sampler import Sampler
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
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
# from skimage.measure import compare_ssim, compare_psnr, compare_mse
from torchvision.utils import make_grid


from sklearn.preprocessing import Normalizer, StandardScaler

from dataset import getDataset
from models import LeNetZhu, ResNet18, ResNet34, DnCNN, Gradinversion_lenet, Gradinversion_res18, getModel, Generator
import torchvision.transforms.functional as F
import cv2
import ssl

from utils import compress_one, compress_onebyone, compress_all, getClassIdx, BatchSampler

ssl._create_default_https_context = ssl._create_unverified_context


# sImages = random.sample(range(0,13233), 1)  # The images we want to recover 10002, 54354, 25, 22
# 0.001: 10001, 898,

# 图片格式转换
def image_to_tensor(image, shape_img, device):
    transform1 = transforms.Compose([
        transforms.CenterCrop((shape_img)),  # 只能对PIL图片进行裁剪
        transforms.ToTensor(),
    ])
    dummy_data = transform1(image)
    dummy_data = torch.unsqueeze(dummy_data, 0)
    dummy_data = dummy_data.to(opt.device).requires_grad_(True)
    return dummy_data


# def compress(gradients, compress_rate):
#     dy_dx, indices = [],[]
#     for g in gradients:
#         g1, indice = compresstion(g, compress_rate)
#         g2 = decompress(g1, indice)
#         b = torch.ones(g2.shape)
#         indice = torch.where(g2 == 0, g2, b)
#         # indice = g.ge(g1[0])
#         dy_dx.append(g2)
#         indices.append(indice)
#     return dy_dx, indices

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def getmin_gloss(auxdst, net, criterion, mask_tuple, original_dy_dx, label_pred, transform):
    x = transform(auxdst[0][0]).to(opt.device)
    x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
    best_img = x
    min_gloss = 1000000
    for i in range(len(auxdst)):
        x = transform(auxdst[i][0]).to(opt.device)
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        y_hat = net(x)
        y_loss = criterion(y_hat, label_pred)
        current_dy_dx = torch.autograd.grad(y_loss, net.parameters(), create_graph=True)
        current_dy_dx = list(current_dy_dx)
        j = 0
        for tmp_1, tmp_2 in zip(current_dy_dx, mask_tuple):
            current_dy_dx[j] = torch.mul(tmp_1, tmp_2.to(opt.device))
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


def HCGLA_generator_batch(dy_dx, label, generator, num_classes):
    fullconnectGrad = dy_dx[-2].reshape(num_classes, 768, 1, 1).to(opt.device)
    grad_input = fullconnectGrad[[int(t) for t in label], :]
    recons = generator(grad_input)
    return recons


def grad2InitImg(attacked_model, gi_model, original_img, attacked_model_type="res18"):
    attacked_model.to(opt.device)
    gi_model.to(opt.device)
    original_img = original_img.view(-1, 3, 32, 32)
    pred = attacked_model(original_img.view(-1, 3, 32, 32))
    loss = torch.nn.CrossEntropyLoss()(pred, torch.LongTensor([0]).to(opt.device))
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
        lenet = LeNetZhu(num_classes=20, num_channels=3).to(opt.device)
        gi_lenet = Gradinversion_lenet().to(opt.device)
        gi_lenet.load_state_dict(torch.load("./models/generators/gi_lenet_epoch500_2000img(formal_class_20).pkl"))
        recons = grad2InitImg(lenet, gi_lenet, original_img, attacked_model_type="lenet")
        return recons
    elif attacked_model_type == "res18":
        resnet18 = ResNet18(num_classes=10).to(opt.device)
        gi_res18 = Gradinversion_res18().to(opt.device)
        gi_res18.load_state_dict(torch.load("./models/generators/gi_res18_epoch1500_1500img(formal_class_10).pkl"))
        recons = grad2InitImg(resnet18, gi_res18, original_img, attacked_model_type="res18")
        return recons
    else:
        print("unknown model type")
        exit()


# imgs1->tensor, imgs2->tensor
# imgs1:attacked images, imgs2:generator images
def mse_psnr_ssim(imgs1, imgs2):
    mses = 0.0000000000000000001
    psnrs = 0.0000000000000000001
    ssims = 0.0000000000000000001
    for i in range(len(imgs1)):
        dummy_data_np = imgs2[i].detach().cpu().numpy().transpose(1, 2, 0)
        gt_data_np = imgs1[i].detach().cpu().numpy().transpose(1, 2, 0)
        init_data = dummy_data_np.astype(np.float64)
        gt_data_np = gt_data_np.astype(np.float64)
        mse = mean_squared_error(init_data, gt_data_np)
        psnr = 10 * np.log10(255 * 255 / mse)
        ssim = structural_similarity(init_data, gt_data_np, multichannel=True)
        mses = mses + mse/len(imgs1)
        psnrs = psnrs + psnr/len(imgs1)
        ssims = ssims + ssim/len(imgs1)
    return mses, psnrs, ssims


def recovery(opt, dy_dx, mask_list, gt_data, label, net, criterion, generator, num_classes, batch_num):
    # if opt.filter and opt.filter_method == "HCGLA-Filter":
    #     if opt.model_type == "LeNet":
    #         if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
    #             opt.filter_path = "./models/DenoisingModel/LeNet_MNIST_Filter.pth"
    #             denoise_layer = 48
    #         else:
    #             opt.filter_path = "./models/DenoisingModel/LeNetFilter.pth"
    #             denoise_layer = 17
    #     else:
    #         if opt.dataset == "MNIST" or opt.dataset == "FMNIST":
    #             opt.filter_path = "./models/DenoisingModel/ResNet_MNIST_Filter.pth"
    #             denoise_layer = 17
    #         else:
    #             opt.filter_path = "./models/DenoisingModel/net_799.pth"  # net_799.pth
    #             denoise_layer = 48
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    save_filename = '{}{}_{}_{}_{}_{}_{}'.format(opt.save_path, opt.model_type, opt.method, opt.dataset, opt.compress_rate,
                                           opt.init_method, opt.batchsize)
    # print('%s - %s - Filter:%s' % (opt.method, opt.init_method, str(opt.filter)))
    if not os.path.exists(save_filename):
        os.mkdir(save_filename)
    save_filename = save_filename + "/batch_{}_{}".format(batch_num, batchs[batch_num])
    if not os.path.exists(save_filename):
        os.mkdir(save_filename)
    if opt.init_method == "Random":
        dummy_data = torch.randn(gt_data.size()).to(opt.device).requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        # 产生 输出层大小的label,这的label也是一个向量，而不是具体的某个类
        print("生成随机噪点初始化")
    elif opt.init_method == "generator":
        if opt.model_type == "LeNet":
            dummy_data = HCGLA_generator_batch(dy_dx, label, generator, num_classes)
        else:
            print("not trained this")
            exit()
        dummy_data = torch.tensor(dummy_data.cpu().detach().numpy()).to(opt.device).requires_grad_(True)
        init_data = copy.deepcopy(dummy_data)
        print("生成器初始化完成")
    else:
        print("Someting is wrong with the initial method, try it again!")
        return 0
    # 定义优化器
    if opt.method == 'DLG':
        dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(opt.device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=opt.lr)
    if opt.method == 'geiping':
        label_pred = label
        optimizer = torch.optim.Adam([dummy_data, ], lr=opt.lr)
        print("预测的标签是:", label_pred)
    else:
        label_pred = label
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
        print("预测的标签是:", label_pred)

    history = []
    history_iters = []
    losses = []
    mses = []
    psnrs = []
    ssims = []
    train_iters = []
    print('lr =', opt.lr)
    for iters in range(opt.Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if opt.method == 'DLG':
                # 将假的预测进行softmax归一化，转换为概率
                dummy_loss = - torch.mean(
                    torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            else:
                dummy_loss = criterion(pred, label_pred.long())
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            # dummy_dy_dx的非topk位置置0
            dummy_dy_dx = list(dummy_dy_dx)
            # dummy_dy_dx = list((_.detach().clone().cpu() for _ in dummy_dy_dx))
            if opt.method == 'HCGLA':
                i = 0
                for tmp_1, tmp_2 in zip(dummy_dy_dx, mask_list):
                    dummy_dy_dx[i] = torch.mul(tmp_1, tmp_2.to(opt.device))
                    i += 1
                # dummy_dy_dx[-2] = compress(dummy_dy_dx[-2], opt.compress_rate)
            grad_diff = 0
            if opt.method == 'geiping':
                # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                # if opt.model_type == "LeNet":
                ex = dy_dx[0]
                weights = torch.arange(len(dy_dx), 0, -1, dtype=ex.dtype, device=ex.device)/ len(dy_dx)
                for ii in range(len(dy_dx)):
                    grad_diff += 1 - torch.cosine_similarity(dummy_dy_dx[ii].flatten(), dy_dx[ii].flatten(), 0, 1e-10) * weights[ii]
                    grad_diff += ((dummy_dy_dx[ii] - dy_dx[ii]) ** 2).sum()
                grad_diff += total_variation(dummy_data)
            elif opt.method == 'HCGLA':
                if opt.model_type == "LeNet":
                    # gt_fullconnectGrad = dy_dx[-2].reshape(num_classes, 768, 1, 1).to(device)
                    # gt_grad_label = gt_fullconnectGrad[[int(t) for t in label], :]
                    # dummy_fullconnectGrad = dummy_dy_dx[-2].reshape(num_classes, 768, 1, 1).to(device)
                    # dummy_grad_label = dummy_fullconnectGrad[[int(t) for t in label], :]
                    # grad_diff += ((gt_grad_label - dummy_grad_label.to(device)) ** 2).sum()
                    for gx, gy in zip(dummy_dy_dx, dy_dx):
                        grad_diff += ((gx - gy.to(opt.device)) ** 2).sum()
                else:
                    gx = dummy_dy_dx[-2]
                    gy = dy_dx[-2]
                    grad_diff += ((gx - gy) ** 2).sum()
            else:
                for gx, gy in zip(dummy_dy_dx, dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
            # print("grad_diff:", grad_diff)
            grad_diff.backward()
            return grad_diff
        mse, psnr, ssim = mse_psnr_ssim(gt_data, dummy_data)
        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
        print(current_time, iters,
              'loss = %.8f, mse = %.8f, psnr:%.8f, ssim:%.8f' % (current_loss, mses[-1], psnrs[-1], ssims[-1]))

        # if iters <= 30 and iters >= 1 or iters in [40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        if opt.isSave and iters > 0 and (iters % math.ceil(opt.Iteration / 42) == 0):
            history.append([tp(dummy_data[i]) for i in range(opt.batchsize)])
            history_iters.append(iters)

        # if opt.filter and iters % opt.filter_frequency == 0 and iters > 0:
        #     # for k in range(opt.batchsize):
        #     # dummy_img = tp(dummy_data[0].cpu())
        #     # dummy_name = '%s/%s_%s_%s_%s_%s_on_%s_%05d_end.jpg' % (
        #     #     save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.filter_method,
        #     #     opt.dataset, k)
        #     # dummy_img.save(dummy_name)
        #
        #     # HLA去噪
        #     if opt.filter_method == "HCGLA-Filter":
        #         filternet = DnCNN(channels=3, num_of_layers=denoise_layer)
        #         filternet = nn.DataParallel(filternet, device_ids=[0]).to(opt.device)
        #         filternet.load_state_dict(torch.load(opt.filter_path))
        #         filternet = filternet.module
        #         filternet.eval()
        #         # img = Image.open(dummy_name)
        #         # img = tt(img)
        #         # img = img.view(1, *img.size()).to(device)
        #         with torch.no_grad():
        #             Out = torch.clamp(dummy_data - filternet(dummy_data).to(opt.device), 0., 255.)
        #         img = torch.tensor(Out.cpu().detach().numpy())
        #     elif opt.filter_method == 'mean':
        #         pass
        #     # 高斯滤波
        #     elif opt.filter_method == 'Guassian':
        #         pass
        #     # 中值滤波
        #     elif opt.filter_method == 'median':
        #         pass
        #     # 双边滤波
        #     elif opt.filter_method == 'bilater':
        #         pass
        #     else:
        #         print("No such filter method")
        #         exit()
        #     dummy_data = img.to(opt.device)  # .to(device).detach()
        #     dummy_data = dummy_data.requires_grad_(True)
        #     if opt.method == 'DLG':
        #         optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=opt.lr)
        #     if opt.method == 'geiping':
        #         optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
        #     else:
        #         optimizer = torch.optim.LBFGS([dummy_data, ], lr=opt.lr)
        #     if opt.isShow:
        #         history.append([tp(dummy_data[i].cpu()) for i in range(opt.batchsize)])
        #         history_iters.append(iters)
        if current_loss < 0.0000000000000000006:  # converge
            break

    if opt.isSave:
        for i in range(opt.batchsize):
            fig = plt.figure(figsize=(12, 8))
            # if opt.filter:
            #     plt.suptitle(
            #         '%s %s %s %s %s on %s %5d picture' % (opt.model_type, opt.method, opt.init_method, opt.compress_rate,
            #                                           opt.filter_method, opt.dataset, batch_num))
            # else:
            plt.suptitle('%s %s %s %s on %s %5d picture' % (
                opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.dataset, batch_num))
            # fig = plt.figure()
            plt.subplot(4, 11, 1)
            # 绘制真实图像
            plt.imshow(tp(gt_data[i].cpu()))
            plt.axis('off')
            plt.subplot(4, 11, 2)
            plt.imshow(tp(init_data[i]))
            plt.title('Initial', fontdict={"family": "Times New Roman", "size": 16})
            # plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.3, hspace=0.1)
            plt.axis('off')
            for j in range(min(len(history), 42)):
                if j > 0 and history_iters[j] == history_iters[j - 1]:
                    ax1 = fig.add_subplot(4, 11, j + 3)
                    plt.subplot(4, 11, j + 3)
                    plt.imshow(history[j][i])
                    plt.title('DeNoise', fontweight='heavy', color='red',
                              fontdict={"family": "Times New Roman", "size": 16})
                    ax1.spines['top'].set_linewidth('4.0')
                    ax1.spines['right'].set_linewidth('4.0')
                    ax1.spines['bottom'].set_linewidth('4.0')
                    ax1.spines['left'].set_linewidth('4.0')
                    ax1.spines['top'].set_color('red')
                    ax1.spines['right'].set_color('red')
                    ax1.spines['bottom'].set_color('red')
                    ax1.spines['left'].set_color('red')
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                else:
                    plt.subplot(4, 11, j + 3)
                    plt.imshow(history[j][i])
                    plt.title('iter=%d' % (history_iters[j]), fontdict={"family": "Times New Roman", "size": 16})
                    plt.axis('off')
            plt.tight_layout()
            # if opt.filter:
            #     plt.savefig('%s/%s_%s_%s_%s_%s_on_%s_batch%05d_%dth.png' % (
            #         save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.filter_method,
            #         opt.dataset, batch_num, i))
            # else:
            plt.savefig('%s/%s_%s_%s_%s_on_%s_%05d_%dth.png' % (
                save_filename, opt.model_type, opt.method, opt.init_method, opt.compress_rate, opt.dataset, batch_num, i))
            plt.close()

    loss = losses
    # ssim_iDLG = pytorch_ssim.ssim(dummy_data, gt_data).data[0]
    # print('SSIM:', ssim_iDLG)
    print('mse :', mses[-1])
    print('psnr :', psnrs[-1])
    print('ssim :', ssims[-1])
    if mses[-1] < 0.05:
        # avg_ssim += ssim_iDLG
        print('success')
    else:
        print("fail")
    return loss, mses, psnrs, ssims



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_rate', default=0.001, type=float, help='The rate we compress our gradients')

    # about model
    parser.add_argument('--model_type', default='LeNet', choices=["LeNet", "ResNet18"], help='you can choose from LeNet, ResNet18')
    parser.add_argument('--model_path', default='', help=" model path ")
    parser.add_argument('--is_exist', default=False, help='Is the attacked model exist ? or we need to random init it')

    # about data
    parser.add_argument('--dataset', default='CelebA',
                        help='you can choose from CelebA, lfw, pubface, google, MNIST, FMNIST, cifar10, cifar100, ImgNet')

    # init method
    parser.add_argument('--init_method', default='generator',
                        help="you can choose 'Random', 'same_label', 'ori_img', 'RGB', 'img', 'generator'")

    # save path
    parser.add_argument('--save_path', default='./recover_result/image/',
                        help='the path to save recover images')
    parser.add_argument('--isSave', default=True, help='')

    # recovery method
    parser.add_argument('--method', default='HCGLA', help='you can choose from DLG, iDLG, geiping, HCGLA')

    # hyper parameter
    parser.add_argument('--img_path', default='./data/img.jpg', help='')
    parser.add_argument('--lr', default=0.01, type=float, help='')
    parser.add_argument('--Iteration', default=42, type=int, help='')
    parser.add_argument('--device', default="cuda:0", help="the device to run this code")

    # about filter
    # parser.add_argument('--filter', default=True, type=bool, help='')
    # parser.add_argument('--filter_method', default='HCGLA-Filter',
    #                     help="you can choose from 'HCGLA-Filter','mean','Guassian','median','bilater' ")
    # parser.add_argument('--filter_path', default='./models/DenoisingModel/DnCNN/net_799.pth', help='')
    # parser.add_argument('--filter_frequency', default=5, type=int)
    # parser.add_argument('--isShow', default=True, help="")

    # add noise
    parser.add_argument('--noise_level', default=0, type=float, help="")

    # about batch
    parser.add_argument('--batchsize', default=4, choices=[1, 4, 16])
    parser.add_argument('--batchnum', default=2, choices=[])

    # about generator
    parser.add_argument('--generator_path',
                        default="./models/generators")
    opt = parser.parse_args()
    all_losses = []
    all_mses = []
    all_psnrs = []
    all_ssims = []
    transform = transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        transforms.CenterCrop(32),
        transforms.ToTensor()])  #
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        opt.device = 'cuda:0'
    else:
        opt.device = 'cpu'
    print(opt.device)
    if opt.batchsize == 4:
        opt.generator_path = os.path.join(opt.generator_path, "generator_batchsize4.pkl")
    else:
        opt.generator_path = os.path.join(opt.generator_path, "generator_batchsize16.pkl")
    dst, num_classes, channel = getDataset(opt.dataset, opt.model_type)
    net = getModel(num_classes, channel, opt.model_type).to(opt.device)
    if opt.is_exist:
        net.load_state_dict(torch.load(opt.model_path, map_location=opt.device))
    class_idx = getClassIdx(dst)
    batchSampler = BatchSampler(class_idx, opt.batchsize, len(dst), drop_last=True)
    train_loader = torch.utils.data.DataLoader(dst, batch_sampler=batchSampler)
    iterloader = iter(train_loader)
    criterion = nn.CrossEntropyLoss().to(opt.device)
    generator = Generator().to(opt.device)
    generator.load_state_dict(torch.load(opt.generator_path, map_location=torch.device(opt.device)))
    generator.eval()
    Datasetnum = {"CelebA": 202599, "lfw": 13233, "pubface": 9408, "google": 3570, "MNIST": 60000, "FMNIST": 60000,
                  "cifar10": 50000, "cifar100": 50000}
    batchs = [120, 121]# random.sample(range(0, Datasetnum[opt.dataset]//opt.batchsize), opt.batchnum)
    for batch_num in range(opt.batchnum):
        for _ in range(batchs[batch_num]):
            next(iterloader)
        gt_data, label = next(iterloader)
        gt_data = gt_data.to(opt.device)
        label = label.to(opt.device)
        pred = net(gt_data)
        if opt.model_type == "ResNet18" and opt.dataset == "CelebA":
            label = int(label) * 499 // 10176
            label = torch.tensor(np.asarray(label)).to(torch.int64).to(opt.device).view(1)
        elif opt.model_type == "ResNet18" and opt.dataset == "lfw":
            label = int(label) * 499 // 5748
            label = torch.tensor(np.asarray(label)).to(torch.int64).to(opt.device).view(1)
        elif opt.model_type == "ResNet18" and opt.dataset == "ImgNet":
            label = int(label) * 499 // 999
            label = torch.tensor(np.asarray(label)).to(torch.int64).to(opt.device).view(1)
        else:
            label = label
        loss = criterion(pred, label.long())
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
        dy_dx, mask_list = compress_onebyone(grad, opt.compress_rate)
        print("\n\n\n-----------------------%d-----------------------\n" % (batch_num))
        losses, mse, psnr, ssim = recovery(opt, dy_dx, mask_list, gt_data, label, net, criterion, generator, num_classes, batch_num)
        all_losses.append(losses)
        all_mses.append(mse)
        all_psnrs.append(psnr)
        all_ssims.append(ssim)
    # if opt.filter:
    #     filename = "./recover_result/data/" + opt.model_type + "_" + opt.method + "_" + str(
    #         opt.batchsize) + "_" + opt.dataset + "_" + str(opt.filter_method) + "_" + str(opt.compress_rate) + "_" \
    #                + opt.init_method + ".npz"
    # else:
    filename = "./recover_result/data/" + opt.model_type + "_" + opt.method + "_" + str(
        opt.batchsize) + "_" + opt.dataset + "_" + str(opt.compress_rate) + "_" + opt.init_method + ".npz"
    np.savez(filename, all_losses=all_losses, all_mses=all_mses, all_psnrs=all_psnrs, all_ssims=all_ssims)
