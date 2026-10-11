import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Mudra Recognition")

# load the model, encoder and info saved by the notebook
@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("models/mudra_model.keras")
    encoder = joblib.load("models/label_encoder.joblib")
    info = json.load(open("models/model_info.json"))
    return model, encoder, info

model, encoder, info = load_files()
IMG_SIZE = info["img_size"]
class_names = list(encoder.classes_)

page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

# ---------------- Home ----------------
if page == "Home":
    st.title("Bharatanatyam Mudra Recognition")
    st.write("This app recognizes 5 Bharatanatyam hand mudras from an image "
             "and tells how confident the model is.")
    st.subheader("How to use")
    st.write("1. Go to the Predict page")
    st.write("2. Upload a clear photo of your hand")
    st.write("3. See the predicted mudra, the confidence and the probability of every class")
    st.write("Mudras this model can recognize:")
    for name in class_names:
        st.write("- " + name.title())

# ---------------- Predict ----------------
elif page == "Predict":
    st.title("Predict the Mudra")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)

        # same preprocessing as in the notebook: resize and divide by 255
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype="float32") / 255.0
        img = np.expand_dims(img, axis=0)

        probs = model.predict(img)[0]
        best = np.argmax(probs)
        confidence = probs[best] * 100

        st.subheader("Result")
        st.write("Predicted mudra: **" + class_names[best].title() + "**")
        st.write("Confidence: **" + str(round(confidence, 2)) + "%**")

        if confidence < 60:
            st.warning("The model is not very sure. Try a clearer photo with a plain background.")

        st.subheader("Probability of all classes")
        chart_data = pd.DataFrame({"Mudra": [c.title() for c in class_names], "Probability": probs})
        st.bar_chart(chart_data.set_index("Mudra"))
        st.table(chart_data)

# ---------------- About ----------------
else:
    st.title("About")
    st.subheader("Dataset")
    st.write("Total images:", info["total_images"])
    st.write("Train / Validation / Test:", info["train_size"], "/", info["val_size"], "/", info["test_size"])
    st.write("Images in each class:")
    st.bar_chart(pd.Series(info["class_counts"]))

    st.subheader("Model")
    st.write("Model used: " + info["model_name"])
    st.write("Test accuracy: " + str(round(info["test_accuracy"] * 100, 2)) + "%")
    st.write("Input size: " + str(IMG_SIZE) + " x " + str(IMG_SIZE) + " (RGB), pixel values divided by 255")
    st.write("Custom CNN settings: learning rate " + str(info["learning_rate"]) + ", dropout " + str(info["dropout"]))
