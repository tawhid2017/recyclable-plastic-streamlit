import streamlit as st
from PIL import Image
import torch
import torch.serialization  # ✅ Fix starts here
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
import os

# Allow YOLO model class to be unpickled safely
torch.serialization.add_safe_globals([DetectionModel])

st.set_page_config(page_title="Plastic Detection with YOLOv8")

st.title("♻️ Recyclable Plastic Detection with YOLOv8")
st.markdown("Upload an image of plastic waste to detect and classify recyclable types.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load the model safely
    with st.spinner("Running YOLOv8 model..."):
        model = YOLO("best.pt")
        results = model.predict(image, save=False, conf=0.3)

        # Display result
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", use_container_width=True)

        st.success("✅ Detection complete.")

