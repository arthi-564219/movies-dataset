import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

model = load_model('../models/best_model.h5')
class_names = ['Class1', 'Class2', 'Class3']  # Replace with your class names

st.title("🐟 Fish Species Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    pred_idx = np.argmax(prediction)
    confidence = prediction[pred_idx]

    st.write(f"### Prediction: {class_names[pred_idx]}")
    st.write(f"Confidence: {confidence:.2f}")
