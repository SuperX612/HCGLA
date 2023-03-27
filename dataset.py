import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import PIL.Image as Image
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms

from utils import data_augmentation

def normalize(data):
    return data/255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, clear=True):
    # train
    print('process training data')
    files = glob.glob(os.path.join(data_path, 'train', '*.jpg'))
    files.sort()
    if clear:
        h5f = h5py.File('./data/noiseAndclearImage/clear/train.h5', 'w')
    else:
        h5f = h5py.File('./data/noiseAndclearImage/noisy/train.h5', 'w')
    train_num = 0
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(len(files)):
        img = Image.open(files[i])
        Img = img.copy()
        Img = transform(Img)
        # Img = np.float32(normalize(Img))
        h5f.create_dataset(str(train_num), data=Img)
        train_num += 1
        print("file: %s" % (files[i]))
            # patches = Im2Patch(Img, win=patch_size, stride=stride)
            # print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            # for n in range(patches.shape[3]):
            #     data = patches[:,:,:,n].copy()
            #     h5f.create_dataset(str(train_num), data=data)
            #     train_num += 1
            #     for m in range(aug_times-1):
            #         data_aug = data_augmentation(data, np.random.randint(1,8))
            #         h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
            #         train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'test', '*.jpg'))
    files.sort()
    if clear:
        h5f = h5py.File('./data/noiseAndclearImage/clear/test.h5', 'w')
    else:
        h5f = h5py.File('./data/noiseAndclearImage/noisy/test.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = Image.open(files[i])
        # img = np.expand_dims(img, 0)
        img = transform(img)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('test set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f_clear = h5py.File('./data/noiseAndclearImage/clear/train.h5', 'r')
            h5f_noisy = h5py.File('./data/noiseAndclearImage/noisy/train.h5', 'r')
        else:
            h5f_clear = h5py.File('./data/noiseAndclearImage/clear/test.h5', 'r')
            h5f_noisy = h5py.File('./data/noiseAndclearImage/noisy/test.h5', 'r')
        self.keys = list(h5f_clear.keys())
        # random.shuffle(self.keys)
        h5f_clear.close()
        h5f_noisy.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f_clear = h5py.File('./data/noiseAndclearImage/clear/train.h5', 'r')
            h5f_noisy = h5py.File('./data/noiseAndclearImage/noisy/train.h5', 'r')
        else:
            h5f_clear = h5py.File('./data/noiseAndclearImage/clear/test.h5', 'r')
            h5f_noisy = h5py.File('./data/noiseAndclearImage/noisy/test.h5', 'r')
        key = self.keys[index]
        clear_data = np.array(h5f_clear[key])
        noisy_data = np.array(h5f_noisy[key])
        h5f_clear.close()
        h5f_noisy.close()
        return torch.Tensor(noisy_data), torch.Tensor(clear_data)


def CelebA_dataset():
    images_all = []
    labels_all = []
    # print("folders:", folders)
    CelebA_path = "./data/CelebA/Img/img_align_celeba"
    ATTR_DIR = './data/CelebA/Anno/identity_CelebA.txt'
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
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all), transform=transform)
    return dst


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
    lfw_path = './data/lfw'
    folders = os.listdir(lfw_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def pubface_dataset():
    images_all = []
    labels_all = []
    pubface_path = './data/pubface/train'
    folders = os.listdir(pubface_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(pubface_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(pubface_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def google_dataset():
    file = open("./data/google/list_attr.txt", 'r')
    google_path = './data/google/images'
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
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def ImgNet_dataset():
    images_all = []
    labels_all = []
    ImgNet_path = "./data/ilsvrc2012/train"
    folders = os.listdir(ImgNet_path)
    # print("folders:", folders)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(ImgNet_path, fold).replace('\\', '/'))
        for f in files:
            if len(f) > 4 and f[-5:] == '.JPEG':
                images_all.append(os.path.join(ImgNet_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor()])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst


def getDataset(dataname="CelebA", attackedmodel="LeNet"):
    transform = transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        transforms.CenterCrop(32),
        transforms.ToTensor()])  #
    ''' load data '''
    if dataname == 'CelebA':  # classes:10177, counts:202599
        if attackedmodel == "LeNet":
            dst = CelebA_dataset()
            num_classes = 10177
            channel = 3
        else:
            dst = CelebA_dataset()  # CelebA_some_dataset(num=500)
            num_classes = 500
            channel = 3
    elif dataname == 'lfw':  # classes:5749, counts:13233
        if attackedmodel == "LeNet":
            dst = lfw_dataset()
            num_classes = 10177
            channel = 3
        else:
            dst = lfw_dataset()
            num_classes = 500
            channel = 3
    elif dataname == 'pubface':  # classes:120, counts:9408
        dst = pubface_dataset()
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif dataname == 'google':  # classes:475, counts:3570
        dst = google_dataset()
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif dataname == 'cifar10':  # classes:10, counts:50000
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
        dst = datasets.CIFAR10("./data/cifar10", download=False, transform=transform)
    elif dataname == 'cifar100':  # classes:100, counts:50000
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
        dst = datasets.CIFAR100("./data/cifar100", download=False, transform=transform)
    elif dataname == "ImgNet":  # classes:1000, counts:1281167
        dst = ImgNet_dataset()
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif dataname == "MNIST":  # classes:10, counts:60000
        dst = datasets.MNIST("./data/MNIST", download=False, transform=transform)
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    elif dataname == "FMNIST":  # classes:10, counts:60000
        dst = datasets.FashionMNIST("./data/Fashion_MNIST", download=False, transform=transform)
        if attackedmodel == 'LeNet':
            num_classes = 10177
        else:
            num_classes = 500
        channel = 3
    else:
        exit('unknown dataset')
    return dst, num_classes, channel