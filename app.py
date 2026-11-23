import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("flower_classifier.keras")

# -----------------------------
# Class Names
# IMPORTANT:
# Keep the order exactly the same as printed by:
# print(train_generator.class_indices)
# -----------------------------
class_names = [
    "banana_bush",
    "crape_jasmine",
    "dwarf_white_bauhinia",
    "hibiscus",
    "night_flowering_jasmine",
    "rose"
]

# -----------------------------
# Prediction Function
# -----------------------------
# def predict(image):

#     image = image.resize((224, 224))

#     img = np.array(image)

#     img = img / 255.0

#     img = np.expand_dims(img, axis=0)

#     prediction = model.predict(img, verbose=0)

#     predicted_index = np.argmax(prediction)

#     confidence = np.max(prediction)

#     return class_names[predicted_index], confidence

def predict(image):

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    img = np.array(image)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    # 👇 Debugging output
    st.write("Raw Prediction:", prediction)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write("Predicted Index:", predicted_index)
    st.write("Predicted Class:", class_names[predicted_index])

    return class_names[predicted_index], confidence


# -----------------------------
# UI
# -----------------------------
st.title("🌸 Flower Classification System")

st.write(
    "Upload a flower image and the model will predict its species."
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Predicting..."):

            flower, confidence = predict(image)

        st.success(f"Predicted Flower: **{flower.replace('_', ' ').title()}**")

        st.info(f"Confidence: **{confidence * 100:.2f}%**")

        st.progress(float(confidence))

st.markdown("---")

st.markdown(
"""
### Supported Flower Classes

- 🌼 Banana Bush
- 🌼 Crape Jasmine
- 🌼 Dwarf White Bauhinia
- 🌺 Hibiscus
- 🌸 Night Flowering Jasmine
- 🌹 Rose
"""
)