import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image  # Import PIL for image processing

# Load the trained model once to avoid reloading on every prediction
model_path = "trained_plant_disease_model.keras"
model = tf.keras.models.load_model(model_path)

# Function for model prediction
def model_prediction(image):
    image = Image.open(image)  # Open image from BytesIO
    image = image.resize((128, 128))  # Resize to match model input
    input_arr = np.array(image) / 255.0  # Normalize pixel values (0-1)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand batch dimension

    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Get the predicted class index

# Streamlit Sidebar
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display homepage image
img = Image.open("Disease.png")
st.image(img, use_column_width=True)

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")

    # Image Upload
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)  # Display image correctly

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            result_index = model_prediction(test_image)

            # Class names
            class_name = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']
            st.success(f"Model is Predicting it's a {class_name[result_index]}")
            
file_id = "14xRMUvDJj8IbxR86H9y4OoBCk8bts_dF"
url = 'https://drive.google.com/file/d/14xRMUvDJj8IbxR86H9y4OoBCk8bts_dF/view?usp=sharing'
model_path = "trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url,model_path,quiet=False)
