from pathlib import Path
from PIL import Image
import tensorflow as tf
import streamlit as st
from ultralytics import YOLO
from utils import  infer_uploaded_image, infer_uploaded_video

st.title("Butterfly Detection Using Yolov8")
st.sidebar.header("Dashboard")

# model options
task_type = st.sidebar.selectbox("Select Task",["Detection"])

model_type = None
if task_type == "Detection":

    model_type = st.sidebar.selectbox("Select Model",["model"])

else:
    st.error("Currently only 'Detection' function is implemented")

model=None
if model_type=="model":
    model=YOLO("best.pt")
else:
    st.error("Please Select Model in Sidebar")

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) /100

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox("Select Source",["image","video"])

source_img = None
if source_selectbox=="image":
   infer_uploaded_image(confidence, model)

if source_selectbox=="video":
    infer_uploaded_video(confidence, model)
   
