import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
from ultralytics import YOLO  # For YOLOv8
# Import other necessary libraries for Mask-RCNN and EfficientDet

# Function to load the models
@st.cache(allow_output_mutation=True)
def load_yolo_model():
    model = YOLO('yolov8-solar.pt')
    return model

@st.cache(allow_output_mutation=True)
def load_maskrcnn_model():
    # Load your Mask-RCNN model here
    pass

@st.cache(allow_output_mutation=True)
def load_efficientdet_model():
    # Load your EfficientDet model here
    pass

# Load models
yolo_model = load_yolo_model()
# maskrcnn_model = load_maskrcnn_model()
# efficientdet_model = load_efficientdet_model()

# Navigation Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["YOLOv8", "Mask-RCNN", "EfficientDet", "About", "Links"])

# YOLOv8 Page
if page == "YOLOv8":
    st.markdown(
    """
    <style>
    .title-container {
        border: 2px solid #f63366;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("<div class='title-container'><h1>Solar Panel Detection - YOLOv8</h1></div>", unsafe_allow_html=True)
    st.markdown("---")

    file = st.file_uploader("Choose an image", type=["jpg", "png"], key="yolo_uploader")
    
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True, output_format='JPEG')
            
            # Add a border to the image
            st.markdown(
                "<style> img { display: block; margin-left: auto; margin-right: auto; border: 2px solid #ccc; border-radius: 8px; } </style>",
                unsafe_allow_html=True
            )
            
            # YOLOv8 prediction
            results = yolo_model(image)
            # Process and display results
            st.image(results[0].plot(), use_column_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Mask-RCNN Page
elif page == "Mask-RCNN":
    st.markdown(
    """
    <style>
    .title-container {
        border: 2px solid #f63366;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("<div class='title-container'><h1>Solar Panel Detection - Mask-RCNN</h1></div>", unsafe_allow_html=True)
    st.markdown("---")

    file = st.file_uploader("Choose an image", type=["jpg", "png"], key="maskrcnn_uploader")
    
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True, output_format='JPEG')
            
            # Add Mask-RCNN prediction code here
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# EfficientDet Page
elif page == "EfficientDet":
    st.markdown(
    """
    <style>
    .title-container {
        border: 2px solid #f63366;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("<div class='title-container'><h1>Solar Panel Detection - EfficientDet</h1></div>", unsafe_allow_html=True)
    st.markdown("---")

    file = st.file_uploader("Choose an image", type=["jpg", "png"], key="efficientdet_uploader")
    
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True, output_format='JPEG')
            
            # Add EfficientDet prediction code here
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# About Page
elif page == "About":
    st.title("About")
    st.markdown("---")
    st.write("This is a web application that implements three different deep learning models for solar panel detection:")
    st.write("- YOLOv8")
    st.write("- Mask-RCNN")
    st.write("- EfficientDet")
    st.markdown("---")
    st.header("Created by")
    # Add your team information here

elif page == "Links":
    st.title("Links")
    st.markdown("---")
    # Add your relevant links here
