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
from importlib import import_module
import dataset
import model as models

app = FastAPI()

@app.get("/")
def pred(device, model_name, model_path, model_layer:int, transform, img_path):
    device=torch.device(device)
    transform = "pre_"+transform
    model = getattr(models,model_name.split("/")[0])(4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()

    if transform in ["pre_basic_crop", "pre_noise_crop"]:
        w,h = 550, 240
    else:
        w,h = 600, 250
    # model layers
    if model_name.split("/")[0] == "resnet18":
        target_layers = [model.net.layer4[-1].conv2]
    elif model_name.split("/")[0] == "resnet50":
        target_layers = [model.net.layer4[-1].conv3]
    elif model_name.split("/")[0] == "swinT":
        target_layers = [model.net.features[-1][1].norm1]
    elif model_name.split("/")[0] == "swinS":
        target_layers = [model.net.features[-1][1].norm1]
    elif model_name.split("/")[0] == "swinB":
        target_layers = [model.net.features[-1][-1].norm2]

    transform_module = getattr(dataset, transform)
    img = transform_module(img_path)
    pred = int(torch.argmax(torch.nn.Softmax()(model(img))).item()) # 예측값 반환

    image_name = hash(f"{model_name}_{model_path}_{model_layer}_{transform}_{img_path}_image")
    visualization_name = hash(f"{model_name}_{model_path}_{model_layer}_{transform}_{img_path}_visualization")
    image_path = f"./cam_result/knurling_images/{image_name}.jpg"
    visualization_path = f"./cam_result/cam_images/{visualization_name}.jpg"

    if str(image_name) + ".jpg" in os.listdir("./cam_result/knurling_images"):
        return image_path, visualization_path, pred
    else:
        image = cv2.imread(img_path)/255 # 원본 시각화를 위한 image
        image = cv2.resize(image, dsize=(w, h))

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        targets = [ClassifierOutputTarget(pred)] # 
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0,:]
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        visualization = cv2.resize(visualization, dsize=(620,250))

        image = cv2.resize(image, dsize=(620,250)) #해시로 이름값 정하자
        image = image*255
        cv2.imwrite(image_path,image)
        cv2.imwrite(visualization_path,visualization)
        return image_path, visualization_path, pred