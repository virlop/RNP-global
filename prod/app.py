import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import *
import numpy as np
import matplotlib.pyplot as plt


# Configuración de la aplicación
st.title("Segmentación Semántica de Tumores Cerebrales")

# Cargar el modelo
@st.cache_resource
def get_model():
    model = load_model("/workspaces/RNP-global/prod/model.pth")
    model.eval()
    return model

model = get_model()

# Subir archivo
uploaded_file = st.file_uploader("Sube una imagen de resonancia magnética", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Cargar y mostrar imagen original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_container_width=True)
    
    # Preprocesar imagen
    image_tensor = preprocess_single_image(image).unsqueeze(0)
    
    # Realizar la predicción
    with st.spinner("Realizando segmentación..."):
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = postprocess_output(output)
    
    # Visualizar la máscara predicha
    # Convertir la máscara predicha a NumPy y escalarla a 0-255
    pred_mask_np = (pred_mask.numpy() * 255).astype(np.uint8)
    st.image(pred_mask_np, caption="Predicción", use_container_width=True, clamp=True)

    overlay = overlay_mask_on_image(image, pred_mask)
    st.image(overlay, caption="Imagen con máscara superpuesta", use_container_width=True)





            
