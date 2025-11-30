import json
import os
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from .imagecrop import FusionRandomCrop
from torchvision.transforms import functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.bmp','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])

def train_vis_ir_transform():
    return Compose([
		Grayscale(num_output_channels=3),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
		Grayscale(),
        ToTensor()
    ])
def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh

class TrainDataset(Dataset):
    def __init__(self, data_path, exp_path,WIDTH=640, HEIGHT=360):

        self.imgs = open(data_path, 'r').readlines()
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.train_path = os.path.join(exp_path, 'Dataset/Train/')

    def __getitem__(self, index):

        value = self.imgs[index]
        img_names = value.split(' ')

        images_label = (int(img_names[2]) + int(img_names[3]) * 2 + int(img_names[4]) * 3 +
                 int(img_names[5]) * 4 + int(img_names[6][:-1]) * 5 - 1)

        img_1 = cv2.imread(self.train_path + img_names[0].lower())

        height, width = img_1.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))

        img_1 = (img_1 - self.mean_I) / self.std_I
        # img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        img_2 = cv2.imread(self.train_path + img_names[1].lower())
        height, width = img_2.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))

        img_2 = (img_2 - self.mean_I) / self.std_I
        # img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_1, img_2], axis=0)

        return (org_img, images_label)

    def __len__(self):

        return len(self.imgs)


class ValDataset(Dataset):
    def __init__(self, data_path, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.work_dir = os.path.join(data_path, 'Dataset')
        self.pair_list = list(open(os.path.join(self.work_dir, 'Val.txt')))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Val/')

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')
        images_label = (int(pari_id[2]) + int(pari_id[3]) * 2 + int(pari_id[4]) * 3 +
                        int(pari_id[5]) * 4 + int(pari_id[6][:-1]) * 5 - 1)

        # load img1 and img2
        img_1 = cv2.imread(self.img_path + pari_id[0].lower())
        img_2 = cv2.imread(self.img_path + pari_id[1].lower())


        height, width = img_1.shape[:2]

        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))

        img_1 = (img_1 - self.mean_I) / self.std_I
        # img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        height, width = img_2.shape[:2]

        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))
        img_2 = (img_2 - self.mean_I) / self.std_I
        # img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_1, img_2], axis=0)

        org_img = torch.tensor(org_img)

        return (org_img, images_label)

    def __len__(self):

        return len(self.pair_list)


class TestDataset(Dataset):
    def __init__(self, data_path,dataset_dir,dataset_txt, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.work_dir = os.path.join(data_path, dataset_dir)
        self.pair_list = list(open(os.path.join(self.work_dir, dataset_txt)))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Test/')
        # if dataset_dir=='SynDataset' or dataset_dir=='Dataset':
        #     self.img_path = os.path.join(self.work_dir, 'Test_small/')
        # else:
        #     self.img_path = os.path.join(self.work_dir, 'Test/')

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')
        images_label = (int(pari_id[2]) + int(pari_id[3]) * 2 + int(pari_id[4]) * 3 +
                        int(pari_id[5]) * 4 + int(pari_id[6][:-1]) * 5 - 1)

        # load img1 and img2
        img_1 = cv2.imread(self.img_path + pari_id[0].lower())
        img_2 = cv2.imread(self.img_path + pari_id[1].lower())
        a2=self.img_path + pari_id[0]
        a=self.img_path + pari_id[1]
        img_names = [pari_id[0], pari_id[1]]

        height, width = img_1.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))
        img_1 = (img_1 - self.mean_I) / self.std_I
        # img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])


        height, width = img_2.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))
        img_2 = (img_2 - self.mean_I) / self.std_I
        # img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])

        org_img = np.concatenate([img_1, img_2], axis=0)
        org_img = torch.tensor(org_img)

        return (org_img, images_label,img_names)

    def __len__(self):

        return len(self.pair_list)