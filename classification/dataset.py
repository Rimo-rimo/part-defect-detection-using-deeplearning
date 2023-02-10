# Dataset & Data Loader
import cv2
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2


class ProductDataset(Dataset):
    def __init__(self, img_path_list, label_list, args, transforms, train_mode=True):
        
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.args = args
        self.transforms = transforms
        self.train_mode = train_mode
                                                                                                    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = cv2.imread(img_path)
        
        # transforms
        img = globals()[self.transforms](img=img, args=self.args)
        
        # set train or valid mode
        if self.train_mode:
            label = self.label_list[index]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.img_path_list)

def basic(img, args):

    transform = A.Compose([
                    A.Resize(always_apply=False, p=1.0, height=args.height, width=args.weight, interpolation=0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                          ])

    return transform(image=img)["image"]

# def heavy_aug(img, args):

#     transform = A.Compose([
#                     A.Resize(always_apply=False, p=1.0, height=args.height, width=args.weight, interpolation=0),
#                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                     ToTensorV2()
#                           ])

#     return transform(image=img)["image"]