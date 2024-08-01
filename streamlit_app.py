import streamlit as st
from PIL import Image
import requests
import io

# Roboflow API details
api_url = "https://detect.roboflow.com/tomato-leaf-disease-detection-gyozv/1"
api_key = "GgPvEVKHvdTxVtJWqVif"

# Application title and description
st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This web application detects common diseases in tomato leaves using the Roboflow API.*
""")

# File uploader for tomato leaf images
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Convert the image to bytes for the API request
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Make an inference request to Roboflow API
    response = requests.post(
        api_url,
        files={"file": img_byte_arr},
        headers={"Authorization": f"Bearer {api_key}"}
    )

    # Display the original image and results from the API
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if response.status_code == 200:
        detections = response.json()

        # Iterate through detected objects
        st.markdown("### Detected Diseases")
        for prediction in detections['predictions']:
            st.write(f"Disease: {prediction['class']}, Confidence: {prediction['confidence']:.2f}")

    else:
        st.write("Error in detection, please try again.")
