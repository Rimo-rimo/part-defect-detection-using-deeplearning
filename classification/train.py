import warnings
warnings.filterwarnings(action="ignore")
import argparse
from collections import Counter
import os
import random
import pandas as pd
import numpy as np
import cv2
from tqdm.auto import tqdm
import unicodedata
from importlib import import_module

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import wandb
import pdb
import metrics

parser = argparse.ArgumentParser()

"""
python train.py --model resnet50 --wandb True --name basic --aug basic ; 
python train.py --model resnet50 --wandb True --name flip_aug --aug flip_aug ; 
python train.py --model resnet50 --wandb True --name noise_aug --aug noise_aug ; 
python train.py --model resnet50 --wandb True --name clahe_aug --aug clahe_aug ; 
python train.py --model resnet50 --wandb True --name heavy_aug --aug heavy_aug ; 

aug
    basic
    flip_aug
    noise_aug
    clahe_aug
    heavy_aug
"""
parser.add_argument("--weight", default=600, type=int)
parser.add_argument("--height", default=250, type=int)
parser.add_argument("--epochs", default=20,type=int)
parser.add_argument("--batch_size", default=16,type=int)
parser.add_argument("--lr", default=0.0001,type=float)
parser.add_argument("--criterion", default="CrossEntropyLoss")
parser.add_argument("--model", default="resnet50")
parser.add_argument("--aug", default="basic")
parser.add_argument("--num_classes", default=4, type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--name", default="test")
parser.add_argument("--wandb", default="False")
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

print("====="*5)
print(args)
print("====="*5)

# Train Config
CFG = {
    "weight":args.weight,
    "height":args.height,
    "epochs":args.epochs,
    "batch_size":args.batch_size,
    "lr":args.lr,
    "criterion":args.criterion,
    "scheduler":None,

    "name":args.name,
    "device":args.device,
    "model":args.model,
    "aug":args.aug,
    "num_classes":args.num_classes,
    "wandb":args.wandb,
    "seed":args.seed
    }


def set_model_folder(args):
    # Check models folder
    if os.path.isdir("./models"):
        pass
    else:
        os.mkdir("./models")
    
    # Set trained model folder
    cnt = 0
    name_ = args.name
    while True:
        if cnt != 0:
            name_ = args.name + "_" + str(cnt)
        
        if os.path.isdir(f"./models/{name_}"):
            cnt += 1
        else:
            os.mkdir(f"./models/{name_}")
            args.name = name_
            break

    # Set GPU
    device = CFG["device"]
    if torch.cuda.is_available():    
        print('Device:', device)
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    # Set seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set Wandb loging
    if args.wandb == "True":
        wandb.init()
        wandb.run.name = args.name
        wandb.config.update(args)

# Get Data
data_folder = "/home/chicken/project/ABL/data/"
image_folder = "/home/chicken/project/ABL/data/crop_images/"

train = pd.read_csv(os.path.join(data_folder, "train_csv_.csv"))
valid = pd.read_csv(os.path.join(data_folder, "test_csv_.csv"))

train = train[train["label"] != -1]
valid = valid[valid["label"] != -1]

train["image_path"] = image_folder + train["image_path"]
valid["image_path"] = image_folder + valid["image_path"]

# dataset & dataloader
dataset_module = getattr(import_module("dataset"), "ProductDataset")
train_dataset = dataset_module(train["image_path"].tolist(), train["label"].tolist(), args, args.aug, True)
valid_dataset = dataset_module(valid["image_path"].tolist(), valid["label"].tolist(), args, "basic", True) # metric을 얻기위해 train_mode = True

train_loader = DataLoader(train_dataset, batch_size = CFG["batch_size"], shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size = CFG["batch_size"], shuffle=True, num_workers=4)

# modeling
model_module = getattr(import_module("model"), CFG["model"])
model = model_module(num_classes=CFG["num_classes"]).to(args.device)

if CFG["criterion"] == "CrossEntropyLoss":
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr = CFG["lr"])

if CFG["scheduler"] == None:
    scheduler = None

def train(model, optimizer, train_loader, valid_loader, scheduler, device, model_name):
            
    model.to(device)
    best_acc = 0

    # train
    for epoch in range(1,CFG["epochs"]+1):
        model.train()
        running_loss = 0.0
        
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
            
        model.eval()

        valid_loss = 0.0
        correct = 0
        # valid
        with torch.no_grad():
            pred_list = []
            test_list = []
            for img, label in tqdm(iter(valid_loader)):
                img, label = img.to(device), label.to(device)

                pred = model(img)
                valid_loss += criterion(pred, label)
                pred = pred.argmax(dim=1, keepdim=True)
                pred_list.extend(pred.cpu().numpy())
                test_list.extend(label.cpu().numpy())
                
        report = classification_report(test_list, pred_list, output_dict = True)
        label_to_class = ["정상", "이중선", "밀림", "찍힘"]

        
        result = metrics.classification_metrics(test_list, pred_list, label_to_class, args.wandb)
        
        print(f"===================== EPOCH_{epoch} =====================")
        print("ACC : ", result["ACC"])
        print("Precsion : ", result["Precision"])
        print("Recall : ", result["Recall"])
        print("F1-Score : ", result["F1-Score"])

        # if best_acc < result["ACC"]:
        #     best_acc = result["ACC"]
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'./models/{model_name}/{epoch}.pth') #이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')
    return metrics


if __name__ == "__main__":
    set_model_folder(args)
    metrics = train(model, optimizer, train_loader,valid_loader, scheduler, args.device, args.name)

    df = pd.DataFrame(metrics)
    name = CFG["name"]
    df.to_csv(f"./results/{name}.csv", index=False)