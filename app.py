import streamlit as st
from PIL import Image
import torch
import torch.serialization
import os

# ✅ Import YOLO only AFTER allowing safe globals
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
torch.serialization.add_safe_globals([Conv2d, Sequential, Conv])

from ultralytics import YOLO

# ✅ Streamlit Page Setup
st.set_page_config(page_title="Plastic Detection with YOLOv8")
st.title("♻️ Recyclable Plastic Detection with YOLOv8")
st.markdown("Upload an image of plastic waste to detect and classify recyclable types.")

# ✅ File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ✅ Load Image and Predict
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running YOLOv8 model..."):
        model = YOLO("best.pt")
        results = model.predict(image, save=False, conf=0.3)

        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", use_container_width=True)

        st.success("✅ Detection complete.")
