import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms

from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="TrainADnCNN")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=300, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--log_file", type=str, default="./logs", help="")
parser.add_argument("--outf", type=str, default="./models/DenoisingModel", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--clear_path", default="./data/noiseAndclearImage/clear")
parser.add_argument("--noisy_path", default="./data/noiseAndclearImage/noisy")
parser.add_argument("--device", default="cuda:0")
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    opt.outf = os.path.join(opt.outf, "DnCNN")
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    device_ids = [0]
    tp = transforms.Compose([transforms.ToPILImage()])
    model = nn.DataParallel(net, device_ids=device_ids).to(opt.device)
    # model.load_state_dict(torch.load('./logs/DnCNN-B/net.pth', map_location='cpu'))
    criterion.to(opt.device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.log_file)
    step = 0
    # noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, (noisy, clear) in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            # img_train = data
            # if opt.mode == 'S':
            #     noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            # if opt.mode == 'B':
            #     noise = torch.zeros(img_train.size())
            #     stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            #     for n in range(noise.size()[0]):
            #         sizeN = noise[0,:,:,:].size()
            #         noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            # imgn_train = img_train + noise
            noise = noisy - clear
            img_train, imgn_train = Variable(clear.to(opt.device)), Variable(noisy.to(opt.device))
            noise = Variable(noise.to(opt.device))
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 255.)
            # plt.subplot(1,3,1)
            # plt.imshow(tp(img_train[0].cpu()))
            # plt.subplot(1,3,2)
            # plt.imshow(tp(imgn_train[0].cpu()))
            # plt.subplot(1,3,3)
            # plt.imshow(tp(out_train[0].cpu()))
            # plt.show()
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                noisy, clear = dataset_val[k]
                noisy, clear = torch.unsqueeze(noisy, 0), torch.unsqueeze(clear, 0),
                img_val, imgn_val = Variable(clear.to(opt.device)), Variable(noisy.to(opt.device))
                out_val = torch.clamp(imgn_val-model(imgn_val), 0., 255.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 255.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_%s.pth'%(epoch)))
    torch.save(model.state_dict(), os.path.join(opt.outf, 'net_final.pth'))


if __name__ == "__main__":
    if torch.cuda.is_available():
        opt.device = 'cuda:0'
    else:
        opt.device = 'cpu'
    print(opt.device)
    if opt.preprocess:
        prepare_data(data_path=opt.clear_path, clear=True)
        prepare_data(data_path=opt.noisy_path, clear=False)
    main()
