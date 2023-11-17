from pathlib import Path
from PIL import Image
import tensorflow as tf
import streamlit as st
from ultralytics import YOLO
from utils import  infer_uploaded_image

@st.cache_resource
def load_model():
    model=YOLO("best.pt")
    return model
#with st.spinner('Model is being loaded..'):
 # model=load_model()

st.title("Butterfly Segmentation Using Yolov8")

st.sidebar.header("DL Moddel config")

# model options
task_type = st.sidebar.selectbox("Select Task",["Detection"])

model_type = None
if task_type == "Detection":

    model_type = st.sidebar.selectbox("Select Model",["model"])

else:
    st.error("Currently only 'Detection' function is implemented")

#model_path = ""
if model_type=="model":
    model=load_model()
else:
    st.error("Please Select Model in Sidebar")

confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 70))


# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",["Image","video"]
    
)

source_img = None
if source_selectbox==Image:
    
    def infer_uploaded_image(conf, model):
        source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    infer_uploaded_image(confidence, model)        



    #model.predict(source = "test_img", show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False, save_crop=False, line_width=2)


