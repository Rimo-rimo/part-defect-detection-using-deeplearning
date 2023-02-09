# Dataset & Data Loader
import cv2

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset


class ItemDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list
                                                                                                    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = cv2.imread(img_path)
        
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
            
        if self.train_mode:
            label = self.label_list[index]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.img_path_list)