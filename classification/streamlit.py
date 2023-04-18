import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import cv2
import os
import streamlit as st
from pycocotools.coco import COCO
import requests


# ===========================parameter===========================
model_folder = "./models/"
image_folder = "../../data/crop_images/"
bbox_image_foler = "../../data/images/"
data_csv = "../../data/test_for_cam-bbox.csv"
json_path = "../../data/data.json"
device = "cpu"
request_ip = "0.0.0.0"
request_port = "30001"
# ===========================parameter===========================

# check state models
model_types = [i for i in os.listdir(model_folder) if i[0]!="."]
model_names = []
for model_type in model_types:
    for model_name in  os.listdir(os.path.join(model_folder, model_type)): 
        if model_name[0] != ".":
            model_names.append(model_type + "/" + model_name)

# get test data
number_to_class = {0:"good", 1:"double", 2:"pull", 3:"crack"}
class_to_number = {"good":0, "double":1, "pull":2, "crack":3}
data = pd.read_csv(data_csv)
data = data[data["label"] != -1]
coco = COCO(json_path)

# streamlit_sidebar
st.markdown("<h1 style='text-align: center; color : #6396D0;'>CAM Dashboard</h1>", unsafe_allow_html=True)
st.title(" ")
with st.sidebar:
    model_name = st.selectbox(":blue[__MODEL NAME__]",model_names, index=0)
    model_epoch = st.slider(":blue[__MODEL EPOCH(*5)__]", min_value=1, max_value=4, value=4)
    model_layer = st.slider(":blue[__LAYER FOR CAM__]", min_value=1, max_value=4, value=4)
    transform = st.selectbox(":blue[__TRANSFORM__]", ["basic", "basic_crop", "noise_crop"], index=2)
    data_class = st.radio(":blue[__DATA CLASS__]", [0,1,2,3], format_func=lambda x:number_to_class[x], index=3)
    data = data[data["label"] == data_class]
    data_index = st.number_input(":blue[__DATA INDEX__]", min_value=0, max_value=len(data)-1, step=1, value=0)

model_path = model_folder + model_name + f"/{model_epoch * 5}.pth"
img_path = image_folder + data["image_path"].iloc[data_index]
gt = number_to_class[data["label"].iloc[data_index]]

image_path, visualization_path, pred = requests.get(f"http://{request_ip}:{request_port}/?device={device}&model_name={model_name}&model_path={model_path}&model_layer={model_layer}&transform={transform}&img_path={img_path}").json()
image = cv2.imread(image_path)
visualization = cv2.imread(visualization_path)

# ==================bounding box================
img = cv2.imread(f"{bbox_image_foler}"+data["image_path"].iloc[data_index])
anns_ids = coco.getAnnIds(imgIds=data["image_id"].iloc[data_index], iscrowd=None)
anns = coco.loadAnns(anns_ids)
for i in anns:
    bbox = i["bbox"]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = x1 + int(bbox[2])
    y2 = y1 + int(bbox[3])
    img = cv2.rectangle(img,(x1,y1),(x2,y2),[255,255,0],10)
img = img[ 700:1500 , 50 :2300]
img = cv2.resize(img, dsize=(850,300))

st.markdown("<h4 style='text-align:center; color:#808080'>[Origin]</h4>", unsafe_allow_html=True)
st.image(img)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4 style='text-align: center; color:#808080'>[Knurling]</h4>", unsafe_allow_html=True)
    st.image(image)
    st.text(f"[GT] : {gt}")
    st.markdown("<h10 style='text-align: center; color : #808080;'>[Origim Image]</h10>", unsafe_allow_html=True)
with col2:
    st.markdown("<h4 style='text-align: center; color:#808080'>[CAM]</h4>", unsafe_allow_html=True)
    st.image(visualization)
    st.text(f"[Pred] : {number_to_class[pred]}")