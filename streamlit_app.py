import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')

# Application title and description
st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This web application detects common diseases in tomato leaves using the YOLO (You Only Look Once) object detection model. The model was trained on a dataset that includes various classes of tomato leaf diseases. The classes are:
- Bacterial Spot
- Early Blight
- Healthy
- Iron Deficiency
- Late Blight
- Leaf Mold
- Leaf Miner
- Mosaic Virus
- Septoria
- Spider Mites
- Yellow Leaf Curl Virus*
""")

# Display a sample image
st.image("path/to/sample_tomato_image.jpg", caption="Sample Tomato Leaf Image")

# File uploader for tomato leaf images
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Predict classes
    results = model(image)
    
    # Save and display results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('results.jpg')  # save image
    
    st.image('results.jpg', caption='Model Prediction')

    # Display disease description and prevention tips
    descriptions = {
        'Bacterial Spot': ("Bacterial Spot causes dark, sunken lesions on tomato leaves. It can lead to significant yield loss if not managed properly.", 
                            "Prevent by using disease-free seeds, practicing crop rotation, and applying copper-based fungicides."),
        'Early Blight': ("Early Blight is characterized by dark, concentric lesions on leaves. It affects the foliage and can reduce fruit quality.", 
                          "Manage by using resistant varieties, applying fungicides, and removing affected plant parts."),
        'Healthy': ("The leaf appears healthy with no visible symptoms of disease.", 
                    "Maintain good agricultural practices to keep plants healthy."),
        'Iron Deficiency': ("Iron Deficiency causes interveinal chlorosis (yellowing between the veins) on leaves. It affects overall plant growth and productivity.", 
                            "Prevent by ensuring adequate iron in the soil through fertilizers or foliar sprays."),
        'Late Blight': ("Late Blight causes dark, water-soaked lesions on leaves and stems, often leading to a rapid decay of the plant.", 
                        "Manage by removing infected plants, applying fungicides, and improving air circulation."),
        'Leaf Mold': ("Leaf Mold results in grayish-green to brown mold on the undersides of leaves, often accompanied by a white, powdery substance.", 
                      "Prevent by providing proper ventilation and avoiding overhead watering."),
        'Leaf Miner': ("Leaf Miner larvae create winding trails or mines in the leaf tissue, causing the leaves to become distorted.", 
                       "Prevent by using insecticides, introducing natural predators, and removing affected leaves."),
        'Mosaic Virus': ("Mosaic Virus causes leaves to display a mottled pattern of light and dark green. It can reduce plant vigor and yield.", 
                         "Prevent by using virus-free seeds and controlling insect vectors like aphids."),
        'Septoria': ("Septoria causes small, dark spots with concentric rings on leaves. It can lead to significant defoliation.", 
                      "Manage by practicing crop rotation, using resistant varieties, and applying fungicides."),
        'Spider Mites': ("Spider Mites cause stippling and discoloration of leaves. They can lead to reduced plant growth and fruit quality.", 
                          "Prevent by maintaining adequate humidity, using miticides, and introducing natural predators."),
        'Yellow Leaf Curl Virus': ("Yellow Leaf Curl Virus causes leaves to curl and turn yellow. It can severely impact plant health and yield.", 
                                   "Prevent by using virus-resistant varieties and controlling insect vectors like whiteflies."),
    }
    
    detected_classes = [r.names[r.cls] for r in results]
    for disease in detected_classes:
        if disease in descriptions:
            st.write(f"**{disease}:** {descriptions[disease][0]}")
            st.write(f"**Prevention Tips:** {descriptions[disease][1]}")
        else:
            st.write(f"**{disease}:** Description not available.")
            st.write("**Prevention Tips:** General tips for maintaining plant health.")

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
    .markdown-text-container {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)
