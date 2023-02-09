import warnings
warnings.filterwarnings(action='ignore')
import argparse
import pandas as pd
import numpy as np
import cv2
import os
import random
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim # 최적화 알고리즘들이 포함힘
from collections import Counter
import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
import wandb

"""
python train.py --name resnet18_128_basic --batch_size 512 --model resnet18 --CoarseDropout_num 1 ; python train.py --name resnet18_128_CD20 --batch_size 512 --model resnet18 --CoarseDropout_num 20 ;  python train.py --name resnet18_128_CD40 --batch_size 512 --model resnet18 --CoarseDropout_num 40 ;  python train.py --name resnet18_128_RB025 --batch_size 512 --model resnet18 --RandomBrightness 0.25 ;  python train.py --name resnet18_128_RB05 --batch_size 512 --model resnet18 --RandomBrightness 0.5 ;  
"""

parser = argparse.ArgumentParser()

parser.add_argument("--weight", default=820, type=int)
parser.add_argument("--height", default=685,type=int)
parser.add_argument("--epochs", default=20,type=int)
parser.add_argument("--batch_size", default=8,type=int)
parser.add_argument("--lr", default=0.0001,type=float)
parser.add_argument("--criterion", default="CrossEntropyLoss")
parser.add_argument("--name", default="test")
parser.add_argument("--device", default="cuda:1")
parser.add_argument("--CoarseDropout_num", default=1,type=int)
parser.add_argument("--CoarseDropout_size", default=15,type=int)
parser.add_argument("--RandomBrightness", default=0.01,type=float)
parser.add_argument("--model", default="resnet50",)

args = parser.parse_args()

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

    "CoarseDropout_num":args.CoarseDropout_num,
    "CoarseDropout_size":args.CoarseDropout_size,
    "RandomBrightness":args.RandomBrightness
    }

wandb.init()
wandb.run.name = CFG["name"]
wandb.config.update(CFG)

# Get Data
data = pd.read_csv("./data/classification_csv.csv")

good = data[data["class"] == 0]
good = good.sample(300)

data = data[data["class"] != 10]
data = data[data["class"] != 0]

data = pd.concat([good,data], axis=0)

# Data Split
train , valid = train_test_split(data, test_size=0.2, random_state=42, stratify=data["class"])

# Dataset & Data Loader
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

w ,h =  CFG["weight"], CFG["height"]
# random_brightness
# rgb shuffle
# 
# train_transform =A.Compose([
#                         A.Resize(always_apply=False, p=1.0,  height=h, width=w, interpolation=0),
#                         A.GaussNoise(always_apply=False, p=0.3, var_limit=(159.3, 204.6)),
#                         A.MotionBlur(always_apply=False, p=0.3, blur_limit=(8, 11)),
#                         A.OneOf([
#                             A.Rotate(always_apply=False, p=1.0, limit=(-14, 14), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
#                             A.HorizontalFlip(always_apply=False, p=1.0),
#                                 ],p=0.5),
#                         A.OneOf([
#                             A.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None, approximate=False),
#                             A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.30, 0.30), shift_limit=(-0.05, 0.05), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
#                             A.RandomResizedCrop(always_apply=False, p=1.0, height=h, width=w, scale=(0.5, 1.0), ratio=(0.75, 1.3), interpolation=0),
#                             A.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=(h, h), height=h, width=w, w2h_ratio=1.0, interpolation=0),
#                             A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
#                                 ],p=0.3),
#                         A.OneOf([
#                             A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True),
#                             A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-11, 11), sat_shift_limit=(-13, 13), val_shift_limit=(-15, 15))
#                                 ],p=0.3),      
#                         A.RandomBrightness(always_apply=False, p=0.5, limit=(-CFG["RandomBrightness"], CFG["RandomBrightness"])),     
#                         A.ISONoise(always_apply=False, p=0.5, intensity=(0.72, 1.06), color_shift=(0.01, 0.05)),  
#                         A.ColorJitter(always_apply=False, p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
#                         A.CoarseDropout(always_apply=False, p=0.5, max_holes=CFG["CoarseDropout_num"], max_height=CFG["CoarseDropout_size"], max_width=CFG["CoarseDropout_size"], min_holes=CFG["CoarseDropout_num"], min_height=CFG["CoarseDropout_size"], min_width=CFG["CoarseDropout_size"]),
#                         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                         ToTensorV2()
#                             ])
train_transform = A.Compose([
                        A.Resize(always_apply=False, p=1.0, height=h, width=w, interpolation=0),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                            ])

test_transform = A.Compose([
                        A.Resize(always_apply=False, p=1.0, height=h, width=w, interpolation=0),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                            ])

train_dataset = ItemDataset(train["image_path"].tolist(), train["class"].tolist(), True, train_transform)
valid_dataset = ItemDataset(valid["image_path"].tolist(), valid["class"].tolist(), True, test_transform)

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

if CFG["model"] == "resnet18":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(in_features = 512, out_features=4, bias=True)
elif CFG["model"] == "resnet50":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = nn.Linear(in_features = 2048, out_features=4, bias=True)

if CFG["criterion"] == "CrossEntropyLoss":
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr = CFG["lr"])

if CFG["scheduler"] == None:
    scheduler = None

# valid_counter = dict(valid["class"].value_counts())

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

        #         correct += pred.eq(label.view_as(pred)).sum().item()
        # valid_acc = 100 * correct / len(valid_loader.dataset)
        # accuracy = accuracy_score(test_list, pred_list)
        # precision = precision_score(test_list, pred_list, average="macro")
        # recall = recall_score(test_list, pred_list, average="macro")
        # f1 = f1_score(test_list, pred_list, average="macro")
        # metrics["Accuracy"].append(accuracy)
        # metrics["Precision"].append(precision)
        # metrics["Recall"].append(recall)
        # metrics["F1-Score"].append(f1)
        # wandb.log({"Accuracy":accuracy, "Precision":precision, "Recall":recall, "F1-Score":f1})
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

        wandb.log(result)

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