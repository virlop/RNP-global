import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
from utils import *

# Configuración de la aplicación
st.title("Segmentación Semántica de Tumores Cerebrales")
st.write("Cargá una imagen de resonancia magnética (MRI) y obtené la segmentación del tumor.")

model = load_model("modelo.pth")

uploaded_file = st.file_uploader("Sube una imagen de resonancia magnética", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Realizar la predicción
    with st.spinner("Realizando segmentación..."):
        with torch.no_grad():
            output = model(image)
        mask = postprocess_output(output)




