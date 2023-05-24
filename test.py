import joblib
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import time

st.title('CSGO Map Classifier')
st.caption('by [Umit Canbolat](https://github.com/hoxsec)')

st.divider()

st.header('Upload an image to predict the map')
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    model_loading_text = st.text("Loading model weights...")
    # Load the saved model weights
    model = joblib.load('model_weights.joblib')
    model_loading_text.text("Model weights loaded.")

    # Load and preprocess the random image
    random_image_path = uploaded_file
    random_image = Image.open(random_image_path)
    st.image(random_image, caption='Raw image', use_column_width=True)
    random_image = random_image.resize((64, 64))  # Resize the image to match the model input size
    random_image = np.array(random_image)
    random_image_flattened = random_image.reshape(1, -1)  # Reshape the image for prediction
    st.image(random_image, caption='Image after processing', use_column_width=False)

    prediction_text = st.text("Predicting...")
    start_time = time.time()
    prediction = model.predict(random_image_flattened)
    final_time = time.time() - start_time
    prediction_text.text("Prediction: " + prediction[0] + "\nTime: " + str(final_time) + " seconds")
    