import warnings
warnings.filterwarnings(action='ignore')
import cv2
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class Result(BaseModel):
    image : str
    visualization : str
    pred : int

@app.get("/")
def pred(device, model_name, model_path, model_layer:int, transform, img_path):
    device=torch.device(device)
    model = globals()[model_name.split("/")[0]](4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()

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
        target_layers = [model.net.features[-1][-1].norm2]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    img = globals()[transform](img_path) # image transform
    pred = int(torch.argmax(torch.nn.Softmax()(model(img))).item()) # 예측값 반환
    targets = [ClassifierOutputTarget(pred)] # 

    image = cv2.imread(img_path)/255 # 원본 시각화를 위한 image
    image = cv2.resize(image, dsize=(w, h))

    grayscale_cam = cam(input_tensor=img, targets=targets)
    grayscale_cam = grayscale_cam[0,:]
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    visualization = cv2.resize(visualization, dsize=(620,250))
    image = cv2.resize(image, dsize=(620,250)) #해시로 이름값 정하자
    image = image*255

    image_name = hash(f"{model_name}_{model_path}_{model_layer}_{transform}_{img_path}_image")
    visualization_name = hash(f"{model_name}_{model_path}_{model_layer}_{transform}_{img_path}_visualization")
    image_path = f"./cam_result/knurling_images/{image_name}.jpg"
    visualization_path = f"./cam_result/cam_images/{visualization_name}.jpg"

    if str(image_name) + ".jpg" in os.listdir("./cam_result/knurling_images"):
        return image_path, visualization_path, pred
    else:
        cv2.imwrite(image_path,image)
        cv2.imwrite(visualization_path,visualization)
        return image_path, visualization_path, pred

# ==================back==================
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