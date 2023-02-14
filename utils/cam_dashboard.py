import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import os
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

# ===========================parameter===========================
model_folder = "/Users/rimo/Documents/paper/detector/classification/models/"
image_folder = "/Users/rimo/Documents/paper/data/crop_images/"
data_csv = "/Users/rimo/Documents/paper/data/test_csv_.csv"
device = torch.device("cpu")
w,h = 600, 250
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

# 특정 이미지를 모델에 입력하기 위한 전처리 함수
def transform_to_tensor(img_path):
    img = cv2.imread(img_path)
    img = Image.fromarray(img[:,:,:3])
    img = torchvision.transforms.Resize((250,600))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))(img)
    img = img.unsqueeze(0)
    return img

class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.net.fc = nn.Linear(in_features = 2048, out_features=num_classes, bias=True)
    def forward(self, x):
        x = self.net(x)
        return x


# ST_사이드바 구축
st.title(':blue[CAM Dashboard]')
with st.sidebar:
    model_name = st.selectbox(":blue[___Select model name___]",model_names, index=0)
    model_epoch = st.slider(":blue[___Select model epoch(*5)___]", min_value=1, max_value=4, value=4)
    model_layer = st.slider(":blue[___Select layer for the CAM___]", min_value=1, max_value=10, value=10)
    transform = st.selectbox(":blue[___Select transform___]", ["basic", "heavy_aug"], index=0)
    data_class = st.radio(":blue[___Select data class___]", [0,1,2,3], format_func=lambda x:number_to_class[x], index=3)
    data = data[data["label"] == data_class]
    data_index = st.number_input(":blue[___Select data index___]", min_value=0, max_value=len(data)-1, step=1, value=0)

# ===========================back===========================
model_path = model_folder + model_name + f"/{model_epoch * 5}.pth"
model = globals()[model_name.split("/")[0]](4)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()

target_layers = [model.net.layer4[-1].conv3]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

img_path = image_folder + data["image_path"].iloc[data_index]
gt = number_to_class[data["label"].iloc[data_index]]
print(" GT  : ",gt)

image = cv2.imread(img_path)/255 # 원본 시각화를 위한 image
image = cv2.resize(image, dsize=(w, h))

img = transform_to_tensor(img_path) # image transform
pred = int(torch.argmax(torch.nn.Softmax()(model(img))).item()) # 예측값 반환
targets = [ClassifierOutputTarget(pred)] # 

grayscale_cam = cam(input_tensor=img, targets=targets)
grayscale_cam = grayscale_cam[0,:]
visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
visualization = cv2.resize(visualization, dsize=(620,250))

image = cv2.resize(image, dsize=(620,250))
print("Pred : ",number_to_class[pred])


# ST_image 시각화
col1, col2 = st.columns(2)
with col1:
    st.subheader("Origin Image")
    st.image(image)
    st.text(gt)

with col2:
    st.subheader("Camed Image")
    st.image(visualization)
    st.text(number_to_class[pred])

