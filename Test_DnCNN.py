import cv2
import os
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--model_dir", type=str, default="./models/DenoisingModel/DnCNN", help='path of log files')
parser.add_argument("--model_name", default="net_final", help="")
parser.add_argument("--test_data_clear", type=str, default='./noiseAndclearImage/test/clear', help='')
parser.add_argument("--test_data_noisy", type=str, default='./noiseAndclearImage/test/noisy', help='')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--device", default="cuda:0")
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    if torch.cuda.is_available():
        opt.device = 'cuda:0'
    else:
        opt.device = 'cpu'
    print(opt.device)
    print('Loading model ...\n')
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(opt.device)
    model.load_state_dict(torch.load(os.path.join(opt.model_dir , opt.model_name + ".pth"), map_location=torch.device(opt.device)))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    clear_files_source = glob.glob(os.path.join('data', opt.test_data_clear, '*.jpg'))
    clear_files_source.sort()
    noisy_files_source = glob.glob(os.path.join('data', opt.test_data_noisy, '*.jpg'))
    noisy_files_source.sort()
    # process data
    psnr_test = 0
    img_num = len(clear_files_source)
    tp = transforms.Compose([transforms.ToPILImage()])
    clear_imgs = []
    noisy_imgs = []
    recovery_imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(img_num):
        # image
        Img = Image.open(clear_files_source[i])
        Img = transform(Img)
        # plt.imshow(tp(torch.Tensor(Img)))
        # plt.show()
        clear_ISource = Img.view(1, *Img.size())
        # noise
        # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        Noisy = Image.open(noisy_files_source[i])
        Noisy = transform(Noisy)
        noisy_ISource = Noisy.view(1, *Noisy.size())
        noise = noisy_ISource - clear_ISource
        ISource, INoisy = clear_ISource.to(opt.device), noisy_ISource.to(opt.device)
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 255.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        clear_imgs.append(tp(ISource[0].cpu()))
        noisy_imgs.append(tp(INoisy[0].cpu()))
        recovery_imgs.append(tp(Out[0].cpu()))
        if len(clear_imgs) == 8:
            plt.figure()
            for j in range(8):
                plt.subplot(3, 8, j+1)
                plt.imshow(clear_imgs[j])
                plt.axis('off')
                plt.subplot(3, 8, 8 + j+1)
                plt.imshow(noisy_imgs[j])
                plt.axis('off')
                plt.subplot(3, 8, 16 + j+1)
                plt.imshow(recovery_imgs[j])
                plt.axis('off')
            plt.savefig("./data/noiseAndclearImage/test/recovery/%s_%s.png"%(opt.model_name, (i+1)//8))
            clear_imgs = []
            noisy_imgs = []
            recovery_imgs = []

        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (clear_files_source[i], psnr))

    if len(clear_imgs) > 0:
        plt.figure()
        for j in range(len(clear_imgs)):
            plt.subplot(3, 8, j+1)
            plt.imshow(clear_imgs[j])
            plt.axis('off')
            plt.subplot(3, 8, 8 + j+1)
            plt.imshow(noisy_imgs[j])
            plt.axis('off')
            plt.subplot(3, 8, 16 + j+1)
            plt.imshow(recovery_imgs[j])
            plt.axis('off')
        plt.savefig("./data/noiseAndclearImage/test/recovery/%s_%s.png"%(opt.model_name, (i+9)//8))

    psnr_test /= img_num
    print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    main()
