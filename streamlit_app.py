import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load the pre-trained YOLO model
model = YOLO('best.pt')  # Load the trained model weights

# Application title and description
st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This web application detects common diseases in tomato leaves using the YOLO (You Only Look Once) object detection model. The model was trained on a dataset that includes various classes of tomato leaf diseases. The classes are:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

# Dictionary for disease descriptions and remedies
disease_info = {
    "Bacterial Spot": {
        "description": "Bacterial spot is a bacterial disease that affects tomato plants, causing spots on leaves, stems, and fruits.",
        "remedy": "Remove infected plants, avoid overhead watering, and use copper-based fungicides."
    },
    "Early Blight": {
        "description": "Early blight is a fungal disease that causes concentric rings on leaves, leading to yellowing and defoliation.",
        "remedy": "Remove affected leaves, rotate crops, and use fungicides."
    },
    "Healthy": {
        "description": "The leaf is healthy with no signs of disease.",
        "remedy": "No action needed. Continue proper care and monitoring."
    },
    "Iron Deficiency": {
        "description": "Iron deficiency causes yellowing between the veins of young leaves.",
        "remedy": "Use iron chelate foliar sprays and improve soil pH."
    },
    "Late Blight": {
        "description": "Late blight is a serious disease that causes dark spots on leaves and fruit, leading to rot.",
        "remedy": "Remove infected plants, avoid wet foliage, and apply fungicides."
    },
    "Leaf Mold": {
        "description": "Leaf mold causes yellow spots on the upper side of leaves and a moldy growth on the underside.",
        "remedy": "Improve air circulation, reduce humidity, and use fungicides."
    },
    "Leaf Miner": {
        "description": "Leaf miners are insect larvae that tunnel through leaves, creating winding, white trails.",
        "remedy": "Remove and destroy affected leaves, use insecticides, and introduce natural predators."
    },
    "Mosaic Virus": {
        "description": "Mosaic virus causes mottled, discolored leaves and stunted plant growth.",
        "remedy": "Remove infected plants, control aphids, and use resistant varieties."
    },
    "Septoria": {
        "description": "Septoria leaf spot causes small, circular spots on leaves, leading to yellowing and defoliation.",
        "remedy": "Remove affected leaves, avoid overhead watering, and use fungicides."
    },
    "Spider Mites": {
        "description": "Spider mites cause tiny yellow spots on leaves and fine webbing, leading to leaf drop.",
        "remedy": "Spray with water to remove mites, use insecticidal soap, and introduce natural predators."
    },
    "Yellow Leaf Curl Virus": {
        "description": "Yellow leaf curl virus causes yellowing and curling of leaves, and stunted plant growth.",
        "remedy": "Remove infected plants, control whiteflies, and use resistant varieties."
    }
}

# File uploader for tomato leaf images
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Predict classes
    results = model(image_array)
    
    # Save and display the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image
        im.save('results.jpg')  # Save the image
    
    st.image('results.jpg', caption='Model Prediction')

    # Display disease information
    detected_classes = set([d['name'] for d in results])
    st.write("### Disease Information and Remedies:")
    for disease in detected_classes:
        if disease in disease_info:
            st.write(f"**{disease}**")
            st.write(f"*Description:* {disease_info[disease]['description']}")
            st.write(f"*Remedy:* {disease_info[disease]['remedy']}")

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
