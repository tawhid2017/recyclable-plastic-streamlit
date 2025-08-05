import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import os

# ✅ Patch to prevent UnpicklingError for YOLOv8 weights
# Patch to safely unpickle YOLOv8 .pt weights in PyTorch ≥ 2.6
from torch.serialization import add_safe_globals
import torch.nn.modules.activation
import torch.nn.modules.batchnorm
import torch.nn.modules.conv
import torch.nn.modules.container
import torch.nn.modules.pooling
import torch.nn.modules.linear
import torch.nn.modules.upsampling
import torch.nn.modules.padding
import torch.nn.modules.normalization

# YOLO-specific modules
from ultralytics.nn.modules.block import C2f,Bottleneck,SPPF
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel

add_safe_globals([
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.container.ModuleDict,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.padding.ConstantPad2d,
    torch.nn.modules.normalization.LayerNorm,
    C2f,
    Conv,
    DetectionModel,
    Bottleneck,
    SPPF,
])


st.set_page_config(page_title="Plastic Detection with YOLOv8")
st.title("♻️ Recyclable Plastic Detection with YOLOv8")
st.markdown("Upload an image of plastic waste to detect and classify recyclable types.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running YOLOv8 model..."):
        model = YOLO("best.pt")
        results = model.predict(image, save=False, conf=0.3)

        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", use_container_width=True)
        st.success("Detection complete.")
