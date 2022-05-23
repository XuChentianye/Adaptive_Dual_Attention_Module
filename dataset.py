import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2 as cv
import random


# To make sure a pair of image & mask are randomly augmented in the same way, 
# here I pack these transform together.
def augment(img, mask):
    # Transform lib is often used to process PIL image.
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    
    img = transforms.Resize((384, 512), interpolation=InterpolationMode.BICUBIC)(img)
    mask = transforms.Resize((384, 512), interpolation=InterpolationMode.BICUBIC)(mask)

    H, W = mask.size[0], mask.size[1]

    if random.random() > 0.5:  # Horizontal flipping
        img = tf.hflip(img)
        mask = tf.hflip(mask)

    if random.random() > 0.5:  # Vertical flipping
        img = tf.vflip(img)
        mask = tf.vflip(mask)
    
    scale_ratio = random.uniform(0.6,1.3)  # Scaling (range from 0.6 to 1.3) and Cropping
    img = transforms.Resize((int(scale_ratio*H), int(scale_ratio*W)))(img)
    mask = transforms.Resize((int(scale_ratio*H), int(scale_ratio*W)))(mask)
    img = transforms.CenterCrop((H, W))(img)
    mask = transforms.CenterCrop((H, W))(mask)

    angle = transforms.RandomRotation.get_params([0, 180])  # Rotations (randomly from 0 to 180 degrees)
    img = img.rotate(angle)
    mask = mask.rotate(angle)

    img = np.array(img)
    mask = np.array(mask)

    return img, mask


def augment2(img, mask):
    # Transform lib is often used to process PIL image.
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    
    img = transforms.Resize((384, 512), interpolation=InterpolationMode.BICUBIC)(img)
    mask = transforms.Resize((384, 512), interpolation=InterpolationMode.BICUBIC)(mask)

    img = np.array(img).transpose(1,0,2)
    mask = np.array(mask).transpose(1,0,2)
    return img, mask


class ISICDataset_Seg(data.Dataset):
    def __init__(self, data_path, mask_path):
        self.images = [data_path + '/' + f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.masks = [mask_path + '/' + f for f in os.listdir(mask_path) if f.endswith('.png')]
        self.masks = sorted(self.masks)
        self._len = len(self.images)
        # check len
        if(len(self.images)!=len(self.masks)):
            print("Error! The length of images and masks don't match!")
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
    def __getitem__(self, item):
        image_pa = self.images[item]
        mask_pa = self.masks[item]
        
        img = cv.imread(image_pa)
        ma = cv.imread(mask_pa)
        img, ma = augment(img, ma)

        img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img3 = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        img1 = self.img_transform(img1)
        img2 = self.img_transform(img2)
        img3 = self.img_transform(img3)
        img1 = np.array(img1)
        img2 = np.array(img2)[1:3,:,:]  # drop the H channel
        img3 = np.array(img3)
        image = np.concatenate([img1, img2, img3], axis=0)
        image = torch.from_numpy(image)
        
        ma = Image.fromarray(ma)
        ma = self.mask_transform(ma)

        ma = np.array(ma)
        mask = torch.from_numpy(ma)[0:1,:,:]

        return image, mask
        
    def __len__(self):
        return self._len

class ISICDataset_Seg_Val(data.Dataset):
    def __init__(self, data_path, mask_path):
        self.images = [data_path + '/' + f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.masks = [mask_path + '/' + f for f in os.listdir(mask_path) if f.endswith('.png')]
        self.masks = sorted(self.masks)
        self._len = len(self.images)
        # check len
        if(len(self.images)!=len(self.masks)):
            print("Error! The length of images and masks don't match!")
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
    def __getitem__(self, item):
        image_pa = self.images[item]
        mask_pa = self.masks[item]
        
        img = cv.imread(image_pa)
        ma = cv.imread(mask_pa)
        img, ma = augment2(img, ma)
        
        img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img3 = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        img1 = self.img_transform(img1)
        img2 = self.img_transform(img2)
        img3 = self.img_transform(img3)
        img1 = np.array(img1)
        img2 = np.array(img2)[1:3,:,:]  # drop the H channel
        img3 = np.array(img3)
        image = np.concatenate([img1, img2, img3], axis=0)
        image = torch.from_numpy(image)
        
        ma = Image.fromarray(ma)
        ma = self.mask_transform(ma)

        ma = np.array(ma)
        mask = torch.from_numpy(ma)[0:1,:,:]

        return image, mask
        
    def __len__(self):
        return self._len
    
label_dict = {'[0. 0.]': 2, '[1. 0.]': 0, '[0. 1.]': 1}
class ISICDataset_Cla(data.Dataset):
    def __init__(self, data_path, label_path):
        self.images = [data_path + '/' + f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.csv = np.loadtxt(label_path, delimiter=',', skiprows=1, usecols=(1, 2))
        self.csv = np.array([label_dict[str(i)] for i in self.csv])
        self._len = len(self.images)
        # check len
        if(len(self.images)!=len(self.csv)):
            print("Error! The length of images and masks don't match!",len(self.images), len(self.csv))
        self.img_transform = transforms.Compose([
            transforms.Resize((384, 512), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        self.augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             ])
        
    def __getitem__(self, item):
        image_pa = self.images[item]
        img = cv.imread(image_pa)
        
        seed = np.random.randint(2147483647)
        random.seed(seed)

        img = Image.fromarray(img)
        img = self.augmentation(img)
        img = np.array(img)
        img = Image.fromarray(img)
        img = self.img_transform(img)
        img = np.array(img)
        image = torch.from_numpy(img)
        label = torch.LongTensor(np.array([self.csv[item]]))[0]
        return image, label
        
    def __len__(self):
        return self._len


# # test 1
# data = ISICDataset_Cla(r'.\original_data\ISIC2017\ISIC-2017_Training_Data',r'.\original_data\ISIC2017\ISIC-2017_Training_Part3_GroundTruth.csv')
# print(len(data))  # test __len__

# # test 2
# print(data[0][0].shape, data[0][1])

# # test 3
# im = Image.fromarray(np.array(data[3][0][2,:,:]))
# im.save("test_mask.tiff")

# # test Seg
# data = ISICDataset_Seg(r'.\original_data\ISIC2017\ISIC-2017_Training_Data',r'.\original_data\ISIC2017\ISIC-2017_Training_Part1_GroundTruth')
# temp = data[2]
# im = Image.fromarray(np.array(temp[0][0,:,:]))
# im.save("test_img.tiff")
# ma = Image.fromarray(np.array(temp[1][0,:,:]))
# ma.save("test_mask.tiff")