import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import os

st.set_page_config(page_title="Plastic Detection with YOLOv8")
st.title("♻️ Recyclable Plastic Detection with YOLOv8")
st.markdown("Upload an image of plastic waste to detect and classify recyclable types.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running YOLOv8 model..."):
        # ✅ Patch: Load best.pt manually with weights_only=False
        model_data = torch.load("best.pt", map_location="cpu", weights_only=False)
        model = YOLO("best.pt")  # Still required to wrap it into YOLO model

        results = model.predict(image, save=False, conf=0.3)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", use_container_width=True)

        st.success("✅ Detection complete.")
