from pathlib import Path
from PIL import Image
import tensorflow as tf
import streamlit as st
from ultralytics import YOLO
from utils import  infer_uploaded_image, infer_uploaded_video

@st.cache_resource
def load_model():
    model=YOLO("best.pt")
    return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.title("Butterfly Segmentation Using Yolov8")

st.sidebar.header("DL Moddel config")

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) /100

# model options
task_type = st.sidebar.selectbox("Select Task",["Detection"])

model_type = None
if task_type == "Detection":

    model_type = st.sidebar.selectbox("Select Model",["model"])

else:
    st.error("Currently only 'Detection' function is implemented")

model_path = ""
if model_type:
    model
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model 
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox("Select Source",["image","video"])
#source_img= 'test_img'
#)

source_img = None
if source_selectbox=="image":
   infer_uploaded_image(confidence, model)

if source_selectbox=="video":
    infer_uploaded_video(confidence, model)
   #pred = model.predict(source = "test_img", show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False, save_crop=False, line_width=2)


