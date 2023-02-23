import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms # 이미지 변환 툴
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
import json
import time

# ===========================parameter===========================
model_folder = "/Users/rimo/Documents/paper/detector/classification/models/"
image_folder = "/Users/rimo/Documents/paper/data/crop_images/"
bbox_image_foler = "/Users/rimo/Documents/paper/data/images/"
data_csv = "/Users/rimo/Documents/paper/data/test_for_cam-bbox.csv"
json_path = "/Users/rimo/Documents/paper/data/data.json"
device = torch.device("cpu")
# ===========================set===========================
model_types = [i for i in os.listdir(model_folder) if i[0]!="."]
model_names = []
for model_type in model_types:
    for model_name in  os.listdir(os.path.join(model_folder, model_type)): 
        if model_name[0] != ".":
            model_names.append(model_type + "/" + model_name)

number_to_class = {0:"good", 1:"double", 2:"pull", 3:"crack"}
class_to_number = {"good":0, "double":1, "pull":2, "crack":3}

data = pd.read_csv(data_csv)
data = data[data["label"] != -1]

coco = COCO(json_path)

# 특정 이미지를 모델에 입력하기 위한 전처리 함수
def transform_to_tensor(img_path):
    img = cv2.imread(img_path)
    img = Image.fromarray(img[:,:,:3])
    img = torchvision.transforms.Resize((250,600))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))(img)
    img = img.unsqueeze(0)
    return img

def basic(img_path):
    img = cv2.imread(img_path)
    w,h = 600,250
    test_transform = A.Compose([
                            A.Resize(always_apply=False, p=1.0, height=h, width=w, interpolation=0),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()
                                ])

    img = test_transform(image=img)["image"]
    img = torch.unsqueeze(img,0)
    return img

def noise_crop(img):
    img = cv2.imread(img)
    transform = A.Compose([
                    A.Resize(always_apply=False, p=1.0, height=250, width=600, interpolation=0),
                    A.CenterCrop(always_apply=False, p=1.0, height=240, width=550),
                    A.OneOf([
                        A.Blur(always_apply=False, p=1.0, blur_limit=(1, 13)),
                        A.AdvancedBlur(always_apply=False, p=1.0, blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
                        A.ISONoise(always_apply=False, p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),         
                             ], p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                          ])
    img = transform(image=img)["image"]
    img = torch.unsqueeze(img,0)
    return img

def basic_crop(img):
    img = cv2.imread(img)
    transform = A.Compose([
                    A.Resize(always_apply=False, p=1.0, height=250, width=600, interpolation=0),
                    A.CenterCrop(always_apply=False, p=1.0, height=240, width=550),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                          ])
    img = transform(image=img)["image"]
    img = torch.unsqueeze(img,0)
    return img

class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.net.fc = nn.Linear(in_features = 512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.net.fc = nn.Linear(in_features = 2048, out_features=num_classes, bias=True)
    def forward(self, x):
        x = self.net(x)
        return x

class swinT(nn.Module):
    def __init__(self, num_classes):
        super(swinT, self).__init__()
        self.net = torchvision.models.swin_t(weights='DEFAULT')
        self.net.head = nn.Linear(in_features = 768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class swinS(nn.Module):
    def __init__(self, num_classes):
        super(swinS, self).__init__()
        self.net = torchvision.models.swin_s(weights='DEFAULT')
        self.net.head = nn.Linear(in_features = 768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x

class swinB(nn.Module):
    def __init__(self, num_classes):
        super(swinB, self).__init__()
        self.net = torchvision.models.swin_b(weights='DEFAULT')
        self.net.head = nn.Linear(in_features = 1024, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.net(x)
        return x


# ST_사이드바 구축
# st.title('[CAM Dashboard]')
st.markdown("<h1 style='text-align: center; color : #0067C8;'>CAM Dashboard</h1>", unsafe_allow_html=True)
st.title(" ")
with st.sidebar:
    model_name = st.selectbox(":blue[__MODEL NAME__]",model_names, index=0)
    model_epoch = st.slider(":blue[__MODEL EPOCH(*5)__]", min_value=1, max_value=4, value=4)
    model_layer = st.slider(":blue[__LAYER FOR CAM__]", min_value=1, max_value=4, value=4)
    transform = st.selectbox(":blue[__TRANSFORM__]", ["basic", "basic_crop", "noise_crop"], index=0)
    data_class = st.radio(":blue[__DATA CLASS__]", [0,1,2,3], format_func=lambda x:number_to_class[x], index=3)
    data = data[data["label"] == data_class]
    data_index = st.number_input(":blue[__DATA INDEX__]", min_value=0, max_value=len(data)-1, step=1, value=0)

# ===========================back===========================
model_path = model_folder + model_name + f"/{model_epoch * 5}.pth"
model = globals()[model_name.split("/")[0]](4)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()

start = time.time()
if transform in ["basic_crop", "noise_crop"]:
    w,h = 550, 240
else:
    w,h = 600, 250
if model_name.split("/")[0] == "resnet18":
    target_layers = [model.net.layer4[-1].conv2]
elif model_name.split("/")[0] == "resnet50":
    if model_layer == 1:
        target_layers = [model.net.layer1[-1].conv3]
    elif model_layer == 2:
        target_layers = [model.net.layer2[-1].conv3]
    elif model_layer == 3:
        target_layers = [model.net.layer3[-1].conv3]
    elif model_layer == 4:
        target_layers = [model.net.layer4[-1].conv3]
elif model_name.split("/")[0] == "swinT":
    target_layers = [model.net.features[-1][1].norm1]
elif model_name.split("/")[0] == "swinS":
    target_layers = [model.net.features[-1][1].norm1]
elif model_name.split("/")[0] == "swinB":
    target_layers = [model.net.features[-1][1].norm1]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

img_path = image_folder + data["image_path"].iloc[data_index]
gt = number_to_class[data["label"].iloc[data_index]]
print(" GT  : ",gt)

image = cv2.imread(img_path)/255 # 원본 시각화를 위한 image
image = cv2.resize(image, dsize=(w, h))
# image = 


img = globals()[transform](img_path) # image transform
pred = int(torch.argmax(torch.nn.Softmax()(model(img))).item()) # 예측값 반환
targets = [ClassifierOutputTarget(pred)] # 

grayscale_cam = cam(input_tensor=img, targets=targets)
grayscale_cam = grayscale_cam[0,:]
visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
visualization = cv2.resize(visualization, dsize=(620,250))

image = cv2.resize(image, dsize=(620,250))
print("Pred : ",number_to_class[pred])
end = time.time()
# ==================bounding box================
img = cv2.imread("/Users/rimo/Documents/paper/data/images/"+data["image_path"].iloc[data_index])
# print(int(data["image_id"].iloc[data_index]))
anns_ids = coco.getAnnIds(imgIds=data["image_id"].iloc[data_index], iscrowd=None)
anns = coco.loadAnns(anns_ids)
print("총 수행 시간 : ", end - start)
for i in anns:
    bbox = i["bbox"]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = x1 + int(bbox[2])
    y2 = y1 + int(bbox[3])
    img = cv2.rectangle(img,(x1,y1),(x2,y2),[255,255,0],10)
print(img.shape)
img = img[ 700:1500 , 50 :2300]
img = cv2.resize(img, dsize=(850,300))
st.markdown("<h4 style='text-align: center;'>[Origin Image]</h4>", unsafe_allow_html=True)
st.image(img)

# ST_image 시각화
# c1, c2, c3 = st.columns(3)
# with c1:
#     st.caption(f":blue[__[MODEL]__] {model_name}/{model_epoch*5}.pth")
# with c2:
#     st.caption(f":blue[[__TRANSFORM__]] {transform}")
# with c3:
#     data_path = data["image_path"].iloc[data_index]
#     st.caption(f":blue[__[DATA]__] {data_path}")
# st.caption(f":blue[Model] -> {model_name}/{model_epoch*5}.pth")
# st.caption(f":blue[Transform] -> {transform}")
# data_path = data["image_path"].iloc[data_index]
# st.caption(f":blue[Data] -> {data_path}")
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4 style='text-align: center;'>[Knurling]</h4>", unsafe_allow_html=True)
    # st.subheader("Origin Image")
    st.image(image)
    st.text(f"[GT] : {gt}")
    # st.markdown("<h10 style='text-align: center; color : #808080;'>[Origim Image]</h10>", unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='text-align: center;'>[CAM Image]</h4>", unsafe_allow_html=True)
    # st.subheader("Camed Image")
    st.image(visualization)
    st.text(f"[Pred] : {number_to_class[pred]}")
    # st.markdown("<h10 style='text-align: center; color : #808080;'>[Origim Image]</h10>", unsafe_allow_html=True)