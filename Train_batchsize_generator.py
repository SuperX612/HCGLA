import argparse
import os
from collections import defaultdict
import random
import numpy as np
import torch
from torchvision import datasets, transforms
import sys

from dataset import CelebA_dataset, getDataset
from models import LeNetZhu, Generator, getModel, Gradinversion_lenet
from utils import compress_one, getClassIdx, BatchSampler

sys.path.append("..")
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import PIL.Image as Image
import copy


def train(grad_model, gi_model, train_loader, optimizer, loss_fn, device, epoch, class_num=10177):
    grad_model.to(device)
    gi_model.to(device)
    global_loss = 0
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        pred = grad_model(x)
        loss = nn.CrossEntropyLoss()(pred, y.long())
        grad = torch.autograd.grad(loss, grad_model.parameters(), retain_graph=True)
        # original_dy_dx = list((_.detach().clone().cpu() for _ in grad))
        dy_dx, _ = compress_one(grad[-2], opt.compress_rate)
        fullconnectGrad = dy_dx.reshape(class_num, 768, 1, 1).to(device)
        grad_input = fullconnectGrad[[int(t) for t in y], :]
        optimizer.zero_grad()
        generate_data = gi_model(grad_input)
        if opt.batchsize == 1:
            generate_data = generate_data.unsqueeze(0)
        reconstruction_loss = loss_fn(generate_data, x)
        reconstruction_loss.backward()
        optimizer.step()
        r_loss = reconstruction_loss.item()
        global_loss += r_loss
        print("{} / {}".format(batch, len(train_loader)))
        print("[ Epoch : {} , iter : {} ], Loss : {:.5f}".format(epoch, batch, r_loss))
    return global_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrainGenerator")
    parser.add_argument("--batchsize", type=int, default=4, choices=[4, 16], help='The batchsize of training data')
    parser.add_argument("--compress_rate", type=int, default=0.001, help="The gradient compression rate")
    parser.add_argument("--reserve", type=bool, default=False, help="Continue train?")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--minloss", default=1000000)
    parser.add_argument("--modelpath", default="", help="If reserve is true, then you must write the model path to continue train")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset", default="CelebA")
    parser.add_argument("--attacked_model", default="LeNet")
    parser.add_argument("--savePath", default="./models/generators")
    opt = parser.parse_args()
    train_dataset, num_classes, channel = getDataset(opt.dataset, opt.attacked_model)
    class_idx = getClassIdx(train_dataset)
    batchSampler = BatchSampler(class_idx, opt.batchsize, len(train_dataset), drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batchSampler)
    # class_num = len(class_idx)
    att_model = getModel(num_classes, channel, opt.attacked_model)
    generator = Generator()
    if opt.reserve:
        generator.load_state_dict(torch.load(opt.modelpath))
    optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    hist = []
    opt.savePath = os.path.join(opt.savePath, f"com{opt.compress_rate}_batchsize{opt.batchsize}")
    if torch.cuda.is_available():
        opt.device = 'cuda:0'
    else:
        opt.device = 'cpu'
    print(opt.device)
    for epoch in range(opt.epochs):
        global_loss = train(att_model, generator, train_loader, optimizer, torch.nn.functional.binary_cross_entropy, opt.device, epoch,
                        class_num=num_classes)
        print(f"epoch: {epoch}, global loss: {global_loss}")
        hist.append(global_loss)
        if global_loss < minloss:
            print("\nsave\n")
            minloss = global_loss
            torch.save(generator.state_dict(),
                       "{}/generator_lr{}_epoch{}_globalLoss{:.5f}.pkl".format(opt.savePath, opt.lr, epoch, global_loss))