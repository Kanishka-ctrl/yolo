import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.yaml')
model = YOLO('best.pt')

# Application title and description
st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This web application detects common diseases in tomato leaves using the YOLO (You Only Look Once) object detection model. The model was trained on a dataset that includes various classes of tomato leaf diseases. The classes are:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

# File uploader for tomato leaf images
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

# Sidebar for additional information
st.sidebar.markdown("## About")
st.sidebar.markdown("GitHub: [Tomato Leaf Disease Detection](https://github.com/your-repo-url)")
st.sidebar.markdown("LinkedIn: [Your LinkedIn](https://www.linkedin.com/your-profile)")
st.sidebar.markdown("## More Information")
st.sidebar.markdown("[GitHub Repository](https://github.com/your-repo-url)")

# Process the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Predict classes
    results = model(image)
    # View results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('results.jpg')  # save image
    
    st.image('results.jpg', caption='Model Prediction')

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
