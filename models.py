# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:01:42 2022

@author: Lomo
"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F


def set_random_seed(seed=233):
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

ngf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(768, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            # nn.Tanh()
            # state size. (nc) x 64 x 64
            nn.MaxPool2d(2, 2)
        )

    def forward(self, input):  # [1, 768, 1, 1]
        input = input.reshape(input.size()[0], input.size()[1], 1, 1)
        return self.main(input)


class Generator_lenet(nn.Module):
    def __init__(self):
        super(Generator_lenet, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(768, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            # nn.Tanh()
            # state size. (nc) x 64 x 64
            nn.MaxPool2d(2, 2)
        )

    def forward(self, input):  # [1, 768, 1, 1]
        return self.main(input)


class Gradinversion_lenet(nn.Module):
    def __init__(self):
        super(Gradinversion_lenet, self).__init__()
        self.generator = Generator_lenet()

    def forward(self, x):  # x: torch.Size([10, 768])
        x = x.reshape(x.size()[0], x.size()[1], 1, 1)
        x = self.generator(x)
        x = torch.mean(x, dim=0)  # torch.Size([3, 32, 32])
        return x


class Generator_res18(nn.Module):
    def __init__(self):
        super(Generator_res18, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(32768, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            # nn.Tanh()
            # state size. (nc) x 64 x 64
            nn.MaxPool2d(2, 2)
        )

    def forward(self, input):  # [1, 32768, 1, 1]
        return self.main(input)


class Gradinversion_res18(nn.Module):
    def __init__(self):
        super(Gradinversion_res18, self).__init__()
        self.generator = Generator_res18()

    def forward(self, x):  # x: torch.Size([10, 32768])
        x = x.reshape(x.size()[0], x.size()[1], 1, 1)
        x = self.generator(x)
        x = torch.mean(x, dim=0)  # torch.Size([3, 32, 32])
        return x


# class Generator(nn.Module):
#     def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.nz = nz
#         self.ngf = ngf
#         self.nc = nc
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#             return output


class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class LeNetZhu(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            set_random_seed(12)
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            set_random_seed(13)
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = torch.sigmoid(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        set_random_seed(12)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(32768, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        set_random_seed(12)
        nn.init.uniform_(layer.weight, a=-0.5, b=0.5)
    elif type(layer) == nn.Linear:
        set_random_seed(12)
        nn.init.uniform_(layer.weight, a=-0.5, b=0.5)
        nn.init.uniform_(layer.bias, a=-0.5, b=0.5)


def ResNet18(num_classes):
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    resnet.apply(init_weights)
    return resnet


def ResNet34(num_classes):
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    resnet.apply(init_weights)
    return resnet


# def ResNet50(num_classes):
#    resnet = ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
#    resnet.apply(init_weights)
#    return resnet

# def ResNet101(num_classes):
#     resnet = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
#     resnet.apply(init_weights)
#     return resnet
#
#
# def ResNet152(num_classes):
#     resnet = ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)
#     resnet.apply(init_weights)
#     return resnet
def getModel(num_classes, channel, modelname="LeNet"):
    if modelname == 'LeNet':
        net = LeNetZhu(num_classes=num_classes, num_channels=channel)
    elif modelname == 'ResNet18':
        net = ResNet18(num_classes=num_classes)
    elif modelname == 'ResNet34':
        net = ResNet34(num_classes=num_classes)
    else:
        print("unknown model-type")
        exit()
    return net


if __name__ == '__main__':
    class_num = 3
    lenet = LeNetZhu(class_num)
    print(lenet.state_dict()["body.0.weight"])
    print(lenet.state_dict()["fc.0.weight"])

    class_num = 4
    lenet = LeNetZhu(class_num)
    print(lenet.state_dict()["body.0.weight"])
    print(lenet.state_dict()["fc.0.weight"])

    res18 = ResNet18(class_num // 2)
    print(res18.state_dict()["conv1.weight"])
    print(res18.state_dict()["linear.weight"])

    res18 = ResNet18(class_num)
    print(res18.state_dict()["conv1.weight"])
    print(res18.state_dict()["linear.weight"])

    test_input = torch.randn(1, 3, 32, 32)

    lenet(test_input)
    res18(test_input)
    res34 = ResNet34(class_num)
    res34(test_input)