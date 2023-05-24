import joblib
from PIL import Image
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import time

st.title('CSGO Map Classifier')
st.caption('Trained on a ~1500 image dataset of various Counter-Strike Global Offensive map images | by [Umit Canbolat](https://github.com/hoxsec)')

st.divider()

sidebar = st.sidebar

model_selector = sidebar.selectbox('Select model size', ['64x64', '256x256', '512x512', '512x512_v2'])
if model_selector == '64x64':
    model = joblib.load('models/model_weights_64.joblib')
    resizer = 64
elif model_selector == '256x256':
    model = joblib.load('models/model_weights_256.joblib')
    resizer = 256
elif model_selector == '512x512':
    model = joblib.load('models/model_weights_512.joblib')
    resizer = 512
elif model_selector == '512x512_v2':
    model = joblib.load('models/model_weights_512_v2.joblib')
    resizer = 512


image_uploader = sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if image_uploader is not None:
    # Load and preprocess the random image
    random_image_path = image_uploader
    random_image = Image.open(random_image_path)
    st.image(random_image, caption='Raw image', use_column_width=True)
    random_image = random_image.resize((resizer, resizer))  # Resize the image to match the model input size
    random_image = np.array(random_image)
    random_image_flattened = random_image.reshape(1, -1)  # Reshape the image for prediction
    st.image(random_image, caption='Image after processing', use_column_width=False)

    prediction_text = st.text("Predicting...")
    start_time = time.time()
    prediction = model.predict(random_image_flattened)
    final_time = time.time() - start_time
    prediction_text.text("Prediction: " + prediction[0] + "\nTime: " + str(final_time) + " seconds")
    