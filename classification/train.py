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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report

torch.backends.cudnn.enabled = False
# import wandb

parser = argparse.ArgumentParser()

"""
python train.py --name test --batch_size 16 --model resnet18 --aug basic 
"""
parser.add_argument("--weight", default=820, type=int)
parser.add_argument("--height", default=685, type=int)
parser.add_argument("--epochs", default=20,type=int)
parser.add_argument("--batch_size", default=8,type=int)
parser.add_argument("--lr", default=0.0001,type=float)
parser.add_argument("--criterion", default="CrossEntropyLoss")
parser.add_argument("--model", default="resnet50")
parser.add_argument("--aug", default="basic")
parser.add_argument("--num_classes", default=4, type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--name", default="test")
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
    }

# loging
# wandb.init()
# wandb.run.name = CFG["name"]
# wandb.config.update(CFG)

# Get Data
data_folder = "/home/chicken/project/ABL/data/"
image_folder = "/home/chicken/project/ABL/data/images/"

train = pd.read_csv(os.path.join(data_folder, "train_csv_.csv"))
valid = pd.read_csv(os.path.join(data_folder, "test_csv_.csv"))

train = train[train["label"] != -1]
valid = valid[valid["label"] != -1]

train["image_path"] = image_folder + train["image_path"]
valid["image_path"] = image_folder + valid["image_path"]

# dataset & dataloader
dataset_module = getattr(import_module("dataset"), "ProductDataset")
train_dataset = dataset_module(train["image_path"].tolist(), train["label"].tolist(), args, args.aug, True)
valid_dataset = dataset_module(valid["image_path"].tolist(), valid["label"].tolist(), args, "basic", True)

train_loader = DataLoader(train_dataset, batch_size = CFG["batch_size"], shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size = CFG["batch_size"], shuffle=True, num_workers=4)

# Modeling
device = CFG["device"]

if torch.cuda.is_available():    
    #device = torch.device("cuda:0")
    print('Device:', device)
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

model_module = getattr(import_module("model"), CFG["model"])
model = model_module(num_classes=CFG["num_classes"]).to(device)

# if CFG["model"] == "resnet18":
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#     model.fc = nn.Linear(in_features = 512, out_features=4, bias=True)
# elif CFG["model"] == "resnet50":
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#     model.fc = nn.Linear(in_features = 2048, out_features=4, bias=True)

if CFG["criterion"] == "CrossEntropyLoss":
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr = CFG["lr"])

if CFG["scheduler"] == None:
    scheduler = None

def train(model, optimizer, train_loader, valid_loader, scheduler, device, model_name):
    cnt = 0
    while True:
        try:
            if cnt ==0:
                os.mkdir(f"./models/{model_name}")
            else:
                os.mkdir(f"./models/{model_name}_{cnt}")
            break
        except:
            pass
        cnt += 1
            
    model.to(device)
    best_acc = 0
    metrics = {"Accuracy":[],"Precision":[],"Recall":[],"F1-Score":[]}
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
        
        # print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))
        
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
        result = dict()

        result["good_precision"] = report["0"]["precision"]
        result["good_F1-score"] = report["0"]["f1-score"]
        result["good_recall"] = report["0"]["recall"]

        result["crack_precision"] = report["1"]["precision"]
        result["crack_F1-score"] = report["1"]["f1-score"]
        result["crack_recall"] = report["1"]["recall"]

        result["pull_precision"] = report["2"]["precision"]
        result["pull_F1-score"] = report["2"]["f1-score"]
        result["pull_recall"] = report["2"]["recall"]

        result["double_precision"] = report["3"]["precision"]
        result["double_F1-score"] = report["3"]["f1-score"]
        result["double_recall"] = report["3"]["recall"]

        result["ACC"] = report["accuracy"]
        result["Precision"] = report["macro avg"]["precision"]
        result["Recall"] = report["macro avg"]["recall"]
        result["F1-Score"] = report["macro avg"]["f1-score"]

        # wandb.log(result)

        valid_loss
        
        print(f"===================== EPOCH_{epoch} =====================")
        print("ACC : ", result["ACC"])
        print("Precsion : ", result["Precision"])
        print("Recall : ", result["Recall"])
        print("F1-Score : ", result["F1-Score"])

        if best_acc < result["ACC"]:
            best_acc = result["ACC"]
            torch.save(model.state_dict(), f'./models/{model_name}/{epoch}.pth') #이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')
    return metrics


if __name__ == "__main__":

    metrics = train(model, optimizer, train_loader,valid_loader, scheduler, device, CFG["name"])

    df = pd.DataFrame(metrics)
    name = CFG["name"]
    df.to_csv(f"./results/{name}.csv", index=False)