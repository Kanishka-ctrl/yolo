import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Load the YOLO model for spot detection
spot_model = YOLO('path_to_spot_detection_model.pt')  # Replace with your spot detection model
# Load the YOLO model for disease classification
disease_model = YOLO('yolov8n.pt')  # Initialize the YOLO model for disease classification

# Application title and description
st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This web application detects common diseases in tomato leaves. It first identifies spots on the leaves and then classifies the type of disease based on these spots. The classes are:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

# File uploader for tomato leaf images
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Step 1: Spot Detection
    spot_results = spot_model(image_array)
    spot_image_array = spot_results[0].plot()  # Get spots highlighted in the image
    spot_image = Image.fromarray(spot_image_array[..., ::-1])  # Convert to RGB PIL image
    spot_image.save('spots.jpg')  # Save the image with spots highlighted

    st.image('spots.jpg', caption='Detected Spots')

    # Step 2: Disease Classification
    # To classify the detected spots, we need to process them individually
    # Here we assume spots are detected and bounding boxes are provided
    st.markdown("### Detected Spots and Their Diseases")
    for box in spot_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])  # Bounding box coordinates
        spot_image_crop = image_array[y1:y2, x1:x2]  # Crop the spot from the image

        # Predict disease based on the cropped spot
        spot_disease_results = disease_model(spot_image_crop)
        for spot_box in spot_disease_results[0].boxes:
            cls = spot_box.cls.numpy()[0]  # Class
            conf = spot_box.conf.numpy()[0]  # Confidence
            st.write(f"Spot Class: {disease_model.names[int(cls)]}, Confidence: {conf:.2f}")

# Custom CSS for a polished look
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    .stFileUploader {
        color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)
