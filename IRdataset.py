import glob
import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import heapq
from collections import Counter
from utils import *



def load_data(path, exts=None):

    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def find_image(base_dir, name, exts):
    
        for ext in exts:
            candidate = os.path.join(base_dir, name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def load_names(root, file_path):
        with open(file_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]

        images, masks = [], []

        for name in names:
            img_path = find_image(os.path.join(root, "images"), name, exts)
            msk_path = find_image(os.path.join(root, "masks"), name, exts)

            if img_path is None:
                raise FileNotFoundError(f"[Image Missing] {name} in images/")
            if msk_path is None:
                raise FileNotFoundError(f"[Mask Missing] {name} in masks/")

            images.append(img_path)
            masks.append(msk_path)

        return images, masks

    train_x, train_y = load_names(path, os.path.join(path, "train.txt"))
    valid_x, valid_y = load_names(path, os.path.join(path, "test.txt"))

    return (train_x, train_y), (valid_x, valid_y)

def load_data1(path):
    def load_images(path):
        image_files = sorted(glob.glob(os.path.join(path, "*.jpg")))
        return image_files

    train_path = os.path.join(path,"train")
    valid_path = os.path.join(path,"val")
    

    train_x = load_images(os.path.join(train_path, "images"))
    train_y = load_images(os.path.join(train_path, "masks"))

    valid_x = load_images(os.path.join(valid_path, "images"))
    valid_y = load_images(os.path.join(valid_path, "masks"))

    return (train_x, train_y), (valid_x, valid_y)

""" Initial mask build using Otsu thresholding. """
def init_mask(images, size):
    def otsu_mask(image, size):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = th.astype(np.int32)
        th = th/255.0
        th = th > 0.5
        th = th.astype(np.int32)
        return img, th

    mask = []
    for image in tqdm(images, total=len(images)):
        name = image.split("/")[-1]
        i, m = otsu_mask(image, size)
        m = rle_encode(m)
        mask.append(m)

    return mask

class IRDATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size =size

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = image.astype(np.float32)


        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples